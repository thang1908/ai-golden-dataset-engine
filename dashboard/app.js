const gallery = document.querySelector('#image-gallery');
const galleryStatus = document.querySelector('#gallery-status');
const galleryView = document.querySelector('#gallery-view');
const detailView = document.querySelector('#detail-view');
const backButton = document.querySelector('#back-button');
const imagePanel = document.querySelector('#result-panel');
const image = document.querySelector('#result-image');
const personKey = document.querySelector('#result-person-key');
const sampleId = document.querySelector('#result-sample-id');
const imagePath = document.querySelector('#result-image-path');
const reviewPanel = document.querySelector('#review-panel');
const noReview = document.querySelector('#no-review');
const reviewForm = document.querySelector('#review-form');
const methodSelect = document.querySelector('#method-select');
const reviewStatus = document.querySelector('#review-status');
const sourceCaptionVerdict = document.querySelector('#source-caption-verdict');
const captionEn = document.querySelector('#caption-en');
const captionVi = document.querySelector('#caption-vi');
const captionNote = document.querySelector('#caption-note');
const captionClaimSummary = document.querySelector('#caption-claim-summary');
const captionClaims = document.querySelector('#caption-claims');
const captionHuman = document.querySelector('#caption-human');
const attributeList = document.querySelector('#attribute-list');
const attributeSummary = document.querySelector('#attribute-summary');
const saveButton = document.querySelector('#save-button');
const exportButton = document.querySelector('#export-button');

const triStateOptions = [
  { value: '', label: 'Không chỉnh' },
  { value: 'true', label: 'Đúng (true)' },
  { value: 'false', label: 'Sai (false)' },
  { value: 'null', label: 'Không xuất hiện (null)' },
];
const captionOptions = triStateOptions.filter(({ value }) => value !== 'null');
let reviews = [];
let selectedReview = null;
let selectedCard = null;

function setStatus(element, message, kind = 'info') {
  element.textContent = message;
  element.dataset.kind = kind;
}

function browserPath(projectPath) {
  return `../${projectPath.split('/').map(encodeURIComponent).join('/')}`;
}

function formatVerdict(value) {
  if (value === true) return 'true · đúng';
  if (value === false) return 'false · sai';
  return 'null · không xuất hiện';
}

function valueForSelect(value) {
  if (value === true) return 'true';
  if (value === false) return 'false';
  if (value === null) return 'null';
  return '';
}

function valueFromSelect(value) {
  if (value === 'true') return true;
  if (value === 'false') return false;
  return null;
}

function addOptions(select, options, current = '') {
  select.replaceChildren(...options.map(({ value, label }) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    option.selected = value === current;
    return option;
  }));
}

function text(value) {
  return value === undefined || value === null || value === '' ? '—' : String(value);
}

function clearReview() {
  reviewPanel.classList.add('is-hidden');
  reviewForm.classList.add('is-hidden');
  noReview.classList.add('is-hidden');
  reviews = [];
  selectedReview = null;
}

function appendClaimRow(container, name, check) {
  const row = document.createElement('article');
  row.className = 'claim-row';
  const title = document.createElement('strong');
  title.textContent = name;
  const mentioned = document.createElement('span');
  mentioned.textContent = check.mentioned ? 'Có nhắc' : 'Không nhắc';
  const verdict = document.createElement('span');
  verdict.className = `verdict verdict-${String(check.is_correct)}`;
  verdict.textContent = formatVerdict(check.is_correct);
  const note = document.createElement('p');
  note.textContent = text(check.note);
  row.append(title, mentioned, verdict, note);
  container.append(row);
}

async function loadReviews(currentSampleId) {
  reviewPanel.classList.remove('is-hidden');
  reviewForm.classList.add('is-hidden');
  noReview.classList.add('is-hidden');
  setStatus(reviewStatus, 'Đang tải evaluation v3…');
  try {
    const response = await fetch(`../api/reviews?sample_id=${encodeURIComponent(currentSampleId)}`, { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || 'Không tải được evaluation v3.');
    reviews = (data.reviews || []).filter(({ evaluation }) => evaluation.status === 'success');
    if (!reviews.length) {
      noReview.classList.remove('is-hidden');
      return;
    }
    methodSelect.replaceChildren(...reviews.map(({ evaluation }) => {
      const option = document.createElement('option');
      option.value = evaluation.method;
      option.textContent = evaluation.method.toUpperCase();
      return option;
    }));
    reviewForm.classList.remove('is-hidden');
    renderReview(reviews[0].evaluation.method);
  } catch (error) {
    noReview.textContent = error.message || 'Không tải được evaluation v3.';
    noReview.classList.remove('is-hidden');
  }
}

function renderReview(method) {
  selectedReview = reviews.find(({ evaluation }) => evaluation.method === method) || null;
  if (!selectedReview) return;
  const { evaluation, override } = selectedReview;
  const caption = evaluation.caption_evaluation || {};
  const attributes = evaluation.caption_attribute_evaluation || {};
  const summary = evaluation.caption_attribute_summary || {};
  methodSelect.value = method;
  sourceCaptionVerdict.textContent = `Gemini: ${formatVerdict(caption.is_correct)}`;
  sourceCaptionVerdict.className = `source-label verdict-${String(caption.is_correct)}`;
  captionEn.textContent = text(evaluation.prediction?.caption);
  captionVi.textContent = text(evaluation.prediction?.caption_vi);
  captionNote.textContent = text(caption.note);
  addOptions(captionHuman, captionOptions, valueForSelect(override?.caption?.is_correct));

  const checks = attributes;
  const mentioned = Object.values(checks).filter((check) => check.mentioned).length;
  const falseClaims = Object.values(checks).filter((check) => check.is_correct === false).length;
  captionClaimSummary.textContent = `${mentioned}/21 thuộc tính được nhắc · ${falseClaims} claim sai`;
  captionClaims.replaceChildren();
  Object.entries(checks).forEach(([field, check]) => appendClaimRow(captionClaims, field, check));

  attributeSummary.textContent = `${summary.mentioned ?? 0} có nhắc · ${summary.correct ?? 0} đúng · ${summary.incorrect ?? 0} sai · ${summary.not_mentioned ?? 0} không nhắc`;
  attributeList.replaceChildren();
  Object.entries(attributes).forEach(([field, value]) => {
    const row = document.createElement('article');
    row.className = 'attribute-row';
    const title = document.createElement('div');
    title.className = 'attribute-name';
    title.innerHTML = `<strong>${field}</strong><span class="verdict verdict-${String(value.is_correct)}">Gemini: ${formatVerdict(value.is_correct)}</span>`;
    const values = document.createElement('div');
    values.className = 'attribute-values';
    values.textContent = `Caption: ${value.mentioned ? 'có nhắc' : 'không nhắc'} · Golden: ${JSON.stringify(evaluation.golden_attributes?.[field])}`;
    const label = document.createElement('label');
    label.textContent = 'Nhãn bạn chỉnh';
    const select = document.createElement('select');
    select.dataset.field = field;
    addOptions(select, triStateOptions, valueForSelect(override?.attributes?.[field]?.is_correct));
    label.append(select);
    const note = document.createElement('p');
    note.className = 'attribute-note';
    note.textContent = text(value.note);
    row.append(title, values, label, note);
    attributeList.append(row);
  });
  setStatus(reviewStatus, override ? 'Đang hiển thị evaluation v3 và phần bạn đã chỉnh.' : 'Đang hiển thị evaluation v3 gốc.', 'success');
}

function selectItem(item, card) {
  selectedCard?.classList.remove('is-selected');
  selectedCard = card;
  selectedCard.classList.add('is-selected');
  galleryView.classList.add('is-hidden');
  detailView.classList.remove('is-hidden');
  imagePanel.classList.remove('is-hidden');
  clearReview();
  personKey.textContent = item.person_key;
  sampleId.textContent = item.sample_id;
  imagePath.textContent = item.image_path;
  image.alt = `Dataset image for person key ${item.person_key}`;
  image.onerror = () => setStatus(galleryStatus, 'Không tải được file ảnh đã map.', 'error');
  image.src = browserPath(item.image_path);
  loadReviews(item.sample_id);
  window.scrollTo({ top: 0, behavior: 'auto' });
}

function renderGallery(items) {
  const ordered = Object.values(items).sort((left, right) => Number(left.sample_id) - Number(right.sample_id));
  gallery.replaceChildren();
  for (const item of ordered) {
    const card = document.createElement('button');
    card.className = 'gallery-card';
    card.type = 'button';
    card.innerHTML = `<img loading="lazy" alt="Dataset sample ${item.sample_id}" src="${browserPath(item.image_path)}"><span>Sample ${item.sample_id}</span>`;
    card.addEventListener('click', () => selectItem(item, card));
    gallery.append(card);
  }
  setStatus(galleryStatus, `${ordered.length} ảnh · Click ảnh để review caption v3.`, 'success');
}

async function loadIndex() {
  setStatus(galleryStatus, 'Đang tải ảnh…');
  try {
    const response = await fetch('../output/dashboard/index.json', { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok || data.schema_version !== '1' || !data.items) throw new Error();
    renderGallery(data.items);
  } catch {
    setStatus(galleryStatus, 'Không tải được index. Hãy chạy python tools/build_dashboard_index.py.', 'error');
  }
}

methodSelect.addEventListener('change', () => renderReview(methodSelect.value));

backButton.addEventListener('click', () => {
  detailView.classList.add('is-hidden');
  galleryView.classList.remove('is-hidden');
  selectedCard?.focus();
  window.scrollTo({ top: 0, behavior: 'auto' });
});

reviewForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!selectedReview) return;
  const attributes = Object.fromEntries(
    [...attributeList.querySelectorAll('select[data-field]')]
      .filter((select) => select.value !== '')
      .map((select) => [select.dataset.field, { is_correct: valueFromSelect(select.value) }]),
  );
  const payload = {
    sample_id: selectedReview.evaluation.sample_id,
    method: selectedReview.evaluation.method,
    caption: captionHuman.value === '' ? null : { is_correct: valueFromSelect(captionHuman.value) },
    attributes,
  };
  saveButton.disabled = true;
  setStatus(reviewStatus, 'Đang lưu nhãn…');
  try {
    const response = await fetch('../api/reviews', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || 'Không thể lưu nhãn.');
    selectedReview.override = data.override;
    setStatus(reviewStatus, 'Đã lưu nhãn đã chỉnh.', 'success');
  } catch (error) {
    setStatus(reviewStatus, error.message || 'Không thể lưu nhãn.', 'error');
  } finally {
    saveButton.disabled = false;
  }
});

exportButton.addEventListener('click', async () => {
  exportButton.disabled = true;
  setStatus(reviewStatus, 'Đang tạo Excel v3…');
  try {
    const response = await fetch('../api/export', { method: 'POST' });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data?.error?.message || 'Không thể xuất Excel v3.');
    }
    const blob = await response.blob();
    const download = document.createElement('a');
    download.href = URL.createObjectURL(blob);
    download.download = 'evaluation_review_v3.xlsx';
    download.click();
    URL.revokeObjectURL(download.href);
    setStatus(reviewStatus, 'Đã xuất Excel v3.', 'success');
  } catch (error) {
    setStatus(reviewStatus, error.message || 'Không thể xuất Excel v3.', 'error');
  } finally {
    exportButton.disabled = false;
  }
});

loadIndex();

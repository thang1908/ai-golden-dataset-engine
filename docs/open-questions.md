# Open questions

None of the questions below blocks the design review itself. OQ-001 and OQ-002
should be resolved by approving or amending this package before implementation.

| ID | Question | Why it matters | Affected design | Blocking | Proposed owner | Needed by | Status |
|---|---|---|---|---|---|---|---|
| OQ-001 | Approve the proposed folder/package names `g1_direct_annotation` through `g5_qa_refinement`, or provide preferred names? | Names become CLI commands, imports, tests, and config prefixes. | Project structure, CLI | Yes for implementation | User | Before scaffold | Open |
| OQ-002 | Should each flow be fully self-contained as proposed, accepting some duplicated C1 adapter code, or may the five folders depend on a new shared core? | Changes independence, maintenance cost, and package boundaries. | Project structure, ADR-001 | Yes for implementation | User | Before scaffold | Open |
| OQ-003 | What consent, retention, residency, and external-provider data-use rules apply to person images and generated annotations? | Images may be personal data and are sent outside the local process. | Security, privacy, operations | No for local prototype; yes before production data | Data/product owner | Before production use | Open |
| OQ-004 | What image volume, provider rate limit, latency target, concurrency target, and cost budget apply? | Determines whether sequential operation remains sufficient and how to tune calls/retries. | Scaling, batch, infrastructure | No for single-image implementation | Product/operations owner | Before batch/production | Open |
| OQ-005 | On G4/G5 loop exhaustion, should the CLI fail as proposed or write the last rejected draft with an explicit non-final status for inspection? | Affects whether every run yields a viewable artifact versus strict final-result semantics. | Output contract, workflow errors | No; proposed strict failure is implementable | User | Before behavior is frozen | Open |
| OQ-006 | For G3, are four fixed evidence-focus prompts acceptable, or should all four calls use identical prompts? | Focus prompts improve diversity without relying on unsupported seed behavior, but make runs non-identical. | G3 prompts and interpretation | No; proposal is fixed focus prompts | User | Before prompt freeze | Open |


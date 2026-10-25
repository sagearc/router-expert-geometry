# Repository rules

## Camera-ready scope

- Keep released code, CLI options, tests, and documentation limited to experiments reported in the paper or submitted rebuttal, plus extensions explicitly requested for the camera-ready revision.
- Verify scope against the submitted text, not merely old code or conversation summaries.
- Do not retain unreported pilots, exploratory diagnostics, unused calibration, or alternate intervention modes behind flags. Keep historical sources and results in the separate audit archive instead.
- Remove dead functions, branches, options, tests, and documentation together. Preserve shared helpers still needed by retained experiments.
- Cleanup must preserve the retained experiment's weights, model settings, evaluation rows, direction construction, projection arithmetic, and ranking. Verify numerical parity on a small case before rerunning.
- Launch reruns from a fresh pinned source snapshot. Inspect existing jobs first, preserve their results, and avoid overlapping duplicate runs.
- Apply this scope to the entire repository, including every experiment, helper, test, dependency, entry point, and Markdown file. Keep one-off cluster launchers and investigation transcripts in the external audit archive, not the release tree.
- Keep numerical discrepancies and limitations in concise validation notes. Cleanup must not hide contrary evidence or turn an incomplete reproduction into a success claim.

## Implementation

- Use concrete types from the pinned model implementation for models,
  layers, routers, experts, hooks, and outputs.
- Do not substitute Any, object, generic modules, or invented protocols
  when the concrete class is known. Use precise unions for supported variants.
- Prefer runtime validation that narrows types. Use casts only at verified
  library boundaries, with an explanation when the reason is non-obvious.
- Add CLI flags only for meaningful user choices. Remove unused,
  redundant, and fixed-value options instead of merely making them optional.
- Read information already present in artifacts from those artifacts.
  Reject conflicting metadata rather than allowing an override.
- Give each shared model or protocol constant one authoritative definition.
  Share constants by meaning, not merely because their values happen to match.
- Keep command examples, validation runners, tests, and metadata consistent
  with CLI and configuration changes.
- Refactoring must preserve numerical behavior, including precision,
  projection chunk sizes, batching, sample order, and RNG consumption.
- Do not add compatibility aliases or fallback paths without a concrete
  supported use case.

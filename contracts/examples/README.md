# Response examples shared by tests

`native-job-succeeded.json` is exactly what `gateway/src/adapters/native.ts` `jobBody()` returns
for a succeeded job (checked by `gateway/test/node/contracts.test.ts`). The benchmark harness
parses it in `benchmarks/tests/test_gateway_shapes.py`, so a change to the gateway's public
response shape breaks both suites instead of silently breaking the benchmark.

`admin-job-attempt.json` is the `attempt` block of `GET /admin/jobs/{id}` (parsed worker
identity + stage measurements of the served attempt), checked by
`gateway/test/workers/api.test.ts` and parsed by the benchmark harness.

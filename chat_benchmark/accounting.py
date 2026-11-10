"""Usage accounting is separate from steady output measurements and billing."""


def account(requests, start_id):
    index = next((i for i, r in enumerate(requests) if r.request_id == start_id), None)
    scoped = requests[index:] if index is not None else []
    fields = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "reasoning_tokens")
    totals = {k: sum(getattr(r, k) for r in scoped) if scoped and all(getattr(r, k) is not None for r in scoped) else None
              for k in fields}
    components = [totals[k] for k in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens")]
    totals["uncached_input_tokens"] = max(0, components[0] - components[1] - components[2]) if all(v is not None for v in components) else None
    costs = [r.client_estimated_usd for r in scoped]
    return {"requests": len(scoped), "tokens": totals,
            "client_estimated_usd": sum(costs) if costs and all(v is not None for v in costs) else None,
            "cost_coverage_requests": sum(v is not None for v in costs), "actual_charged_usd": None,
            "scope": "Inclusive of the prepare request through the last observed request: warm-up, bursts, recorder/status/report calls and other in-chat overhead. Earlier skill-loading and a not-yet-generated final reply are outside this snapshot.",
            "billing_note": "Token usage is client-reported, not an invoice. Native dollar figures, when exposed, are client estimates, not verified charges. Subscription allowance, discounts, credits, regional and long-context rates can differ. No guessed price schedule is applied."}

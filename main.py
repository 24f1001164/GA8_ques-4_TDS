from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

PRIORITY = ["prompt_only", "retrieval", "lora", "qlora"]

CHOOSE_CODES = {
    "INVALID_INPUT",
    "UNAVAILABLE",
    "QUALITY_FLOOR",
    "FRESHNESS_REQUIRED",
    "LATENCY_LIMIT",
    "MEMORY_LIMIT",
    "DATA_LIMIT",
    "COST_LIMIT",
}

REPAIR_CODES = {
    "INVALID_TOKEN",
    "INVALID_PARAMETER",
    "CHAT_TEMPLATE_COUNT",
    "INFERENCE_MODE",
    "FULL_MODEL_ARTIFACT",
    "ADAPTER_FILE_SET",
    "INCOMPLETE_CHECKPOINT",
    "MUTABLE_BASE_REVISION",
    "LINEAGE_MISMATCH",
    "EFFECTIVE_BATCH_MISMATCH",
    "EVAL_LEAKAGE",
    "EVAL_DROPOUT_ACTIVE",
    "RESUME_DIVERGENCE",
}


def is_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= 9007199254740991
    )


def is_positive_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 < x <= 9007199254740991
    )


def is_finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def utf8_sorted_unique(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


# ============================================================
# CHOOSE
# ============================================================

def choose_operation(body):
    policy = body.get("policy")
    candidates = body.get("candidates")

    errors = []

    if not isinstance(policy, dict):
        return invalid_input()

    if not isinstance(candidates, list) or len(candidates) != 4:
        return invalid_input()

    required_policy = [
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests",
    ]

    if any(k not in policy for k in required_policy):
        return invalid_input()

    if not is_finite_number(policy["minQuality"]):
        return invalid_input()

    if not (0 <= float(policy["minQuality"]) <= 1):
        return invalid_input()

    if not isinstance(policy["freshnessRequired"], bool):
        return invalid_input()

    for k in ["maxLatencyMs", "maxMemoryMb", "maxTotalCost"]:
        if not is_finite_number(policy[k]) or float(policy[k]) < 0:
            return invalid_input()

    if not is_safe_int(policy["maxLabeledExamples"]):
        return invalid_input()

    if not is_safe_int(policy["horizonRequests"]):
        return invalid_input()

    by_name = {}

    for c in candidates:
        if not isinstance(c, dict):
            return invalid_input()

        required = [
            "name",
            "available",
            "quality",
            "freshness",
            "latencyMs",
            "memoryMb",
            "labeledExamples",
            "oneTimeCost",
            "recurringCost",
        ]

        if any(k not in c for k in required):
            return invalid_input()

        name = c["name"]

        if (
            not isinstance(name, str)
            or name not in PRIORITY
            or name in by_name
        ):
            return invalid_input()

        if not isinstance(c["available"], bool):
            return invalid_input()

        if (
            not is_finite_number(c["quality"])
            or not 0 <= float(c["quality"]) <= 1
        ):
            return invalid_input()

        if not isinstance(c["freshness"], bool):
            return invalid_input()

        for k in ["latencyMs", "memoryMb", "oneTimeCost", "recurringCost"]:
            if not is_finite_number(c[k]) or float(c[k]) < 0:
                return invalid_input()

        if not is_safe_int(c["labeledExamples"]):
            return invalid_input()

        by_name[name] = c

    # Exactly the four required interventions
    if set(by_name.keys()) != set(PRIORITY):
        return invalid_input()

    eligible = []
    totalCosts = {}
    reasonCodes = {}

    for name in PRIORITY:
        c = by_name[name]
        reasons = []

        total = (
            float(c["oneTimeCost"])
            + float(policy["horizonRequests"]) * float(c["recurringCost"])
        )

        # round to exactly the requested numerical precision
        total = round(total, 12)

        if not c["available"]:
            reasons.append("UNAVAILABLE")

        if float(c["quality"]) < float(policy["minQuality"]):
            reasons.append("QUALITY_FLOOR")

        if policy["freshnessRequired"] and not c["freshness"]:
            reasons.append("FRESHNESS_REQUIRED")

        if float(c["latencyMs"]) > float(policy["maxLatencyMs"]):
            reasons.append("LATENCY_LIMIT")

        if float(c["memoryMb"]) > float(policy["maxMemoryMb"]):
            reasons.append("MEMORY_LIMIT")

        if c["labeledExamples"] > policy["maxLabeledExamples"]:
            reasons.append("DATA_LIMIT")

        if total > float(policy["maxTotalCost"]):
            reasons.append("COST_LIMIT")

        reasons = utf8_sorted_unique(reasons)

        totalCosts[name] = total
        reasonCodes[name] = reasons

        if not reasons:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": totalCosts,
        "reasonCodes": reasonCodes,
    }


# ============================================================
# REPAIR
# ============================================================

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def repair_operation(body):

    reason_codes = set()

    tokens = body.get("tokens")

    # --------------------------------------------------------
    # Tokens / assistant-only loss
    # --------------------------------------------------------

    token_valid = True

    if not isinstance(tokens, list) or len(tokens) == 0:
        token_valid = False
    else:
        for t in tokens:
            if not isinstance(t, dict):
                token_valid = False
                break

            if set(t.keys()) != {"id", "role", "padding", "text"}:
                token_valid = False
                break

            if not is_safe_int(t["id"]):
                token_valid = False
                break

            if t["role"] not in {"system", "user", "assistant"}:
                token_valid = False
                break

            if not isinstance(t["padding"], bool):
                token_valid = False
                break

            if not isinstance(t["text"], str):
                token_valid = False
                break

    if token_valid:
        labels = [
            t["id"] if t["role"] == "assistant" and not t["padding"] else -100
            for t in tokens
        ]
    else:
        labels = [-100] * len(tokens) if isinstance(tokens, list) else []
        reason_codes.add("INVALID_TOKEN")

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    if body.get("templateApplications") != 1:
        reason_codes.add("CHAT_TEMPLATE_COUNT")

    # --------------------------------------------------------
    # Parameters / LoRA
    # --------------------------------------------------------

    parameters = body.get("parameters")
    allowed = body.get("allowedTargets")

    params_valid = True

    if not isinstance(parameters, list):
        params_valid = False

    if not isinstance(allowed, list) or len(allowed) == 0:
        params_valid = False
    else:
        if any(not isinstance(x, str) for x in allowed):
            params_valid = False
        if len(set(allowed)) != len(allowed):
            params_valid = False

    if params_valid:
        seen = set()

        for p in parameters:
            if not isinstance(p, dict):
                params_valid = False
                break

            if set(p.keys()) != {"name", "target", "numel"}:
                params_valid = False
                break

            if not isinstance(p["name"], str) or p["name"] in seen:
                params_valid = False
                break

            seen.add(p["name"])

            if not isinstance(p["target"], str):
                params_valid = False
                break

            if not is_positive_safe_int(p["numel"]):
                params_valid = False
                break

    if not params_valid:
        reason_codes.add("INVALID_PARAMETER")

    trainable = []

    if params_valid:
        for p in parameters:
            if (
                p["target"] in allowed
                and (
                    p["name"].endswith(".lora_A.weight")
                    or p["name"].endswith(".lora_B.weight")
                )
            ):
                trainable.append(p)

        if not trainable:
            reason_codes.add("INVALID_PARAMETER")

    trainable.sort(key=lambda p: p["name"].encode("utf-8"))

    trainable_names = [p["name"] for p in trainable]

    trainable_count = 0

    for p in trainable:
        trainable_count += p["numel"]

        if trainable_count > 9007199254740991:
            reason_codes.add("INVALID_PARAMETER")
            trainable_count = 0
            trainable_names = []
            break

    # --------------------------------------------------------
    # PEFT / inference mode
    # --------------------------------------------------------

    if body.get("inferenceMode") is not False:
        reason_codes.add("INFERENCE_MODE")

    # --------------------------------------------------------
    # Evaluation isolation
    # --------------------------------------------------------

    train_ids = body.get("trainRowIds")
    eval_ids = body.get("evalRowIds")

    eval_valid = True

    if not isinstance(train_ids, list) or not isinstance(eval_ids, list):
        eval_valid = False
    else:
        if len(train_ids) == 0 or len(eval_ids) == 0:
            eval_valid = False

        if any(not isinstance(x, str) or x == "" for x in train_ids):
            eval_valid = False

        if any(not isinstance(x, str) or x == "" for x in eval_ids):
            eval_valid = False

        if len(set(train_ids)) != len(train_ids):
            eval_valid = False

        if len(set(eval_ids)) != len(eval_ids):
            eval_valid = False

        if set(train_ids) & set(eval_ids):
            eval_valid = False

    if not eval_valid:
        reason_codes.add("EVAL_LEAKAGE")

    if body.get("dropoutActiveDuringEval") is not False:
        reason_codes.add("EVAL_DROPOUT_ACTIVE")

    # --------------------------------------------------------
    # Adapter files
    # --------------------------------------------------------

    artifact_files = body.get("artifactFiles")

    expected_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    adapter_pass = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(isinstance(x, str) for x in artifact_files)
        and set(artifact_files) == expected_files
    )

    if not adapter_pass:
        reason_codes.add("ADAPTER_FILE_SET")

    adapter_files = sorted(
        artifact_files if isinstance(artifact_files, list) else [],
        key=lambda x: x.encode("utf-8")
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = body.get("checkpoint")

    checkpoint_keys = {
        "model",
        "optimizer",
        "scheduler",
        "step",
        "rng",
        "dataPosition",
    }

    checkpoint_pass = (
        isinstance(checkpoint, dict)
        and set(checkpoint.keys()) >= checkpoint_keys
    )

    if not checkpoint_pass:
        reason_codes.add("INCOMPLETE_CHECKPOINT")

    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    base_revision = body.get("baseRevision")
    dataset_digest = body.get("datasetDigest")
    code_digest = body.get("codeDigest")
    config_digest = body.get("configDigest")
    expected = body.get("expectedDigests")

    lineage_pass = True

    if not isinstance(base_revision, str) or not HEX40.fullmatch(base_revision):
        reason_codes.add("MUTABLE_BASE_REVISION")
        lineage_pass = False

    for digest in [dataset_digest, code_digest, config_digest]:
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            lineage_pass = False

    if not isinstance(expected, dict):
        lineage_pass = False
    else:
        if expected.get("datasetDigest") != dataset_digest:
            lineage_pass = False
        if expected.get("codeDigest") != code_digest:
            lineage_pass = False
        if expected.get("configDigest") != config_digest:
            lineage_pass = False

    if not lineage_pass:
        reason_codes.add("LINEAGE_MISMATCH")

    # --------------------------------------------------------
    # Effective batch
    # --------------------------------------------------------

    micro = body.get("microBatch")
    accumulation = body.get("gradientAccumulation")
    replicas = body.get("replicas")
    expected_batch = body.get("expectedEffectiveBatch")

    batch_pass = (
        is_positive_safe_int(micro)
        and is_positive_safe_int(accumulation)
        and is_positive_safe_int(replicas)
        and is_positive_safe_int(expected_batch)
    )

    if batch_pass:
        try:
            batch_pass = (
                micro * accumulation * replicas == expected_batch
                and micro * accumulation * replicas <= 9007199254740991
            )
        except Exception:
            batch_pass = False

    if not batch_pass:
        reason_codes.add("EFFECTIVE_BATCH_MISMATCH")

    # --------------------------------------------------------
    # Resume equivalence
    # --------------------------------------------------------

    uninterrupted = body.get("uninterruptedWeights")
    resumed = body.get("resumedWeights")
    tolerance = body.get("resumeTolerance")

    resume_pass = True

    if (
        not isinstance(uninterrupted, list)
        or not isinstance(resumed, list)
        or len(uninterrupted) == 0
        or len(resumed) == 0
        or len(uninterrupted) != len(resumed)
    ):
        resume_pass = False
    else:
        if not all(is_finite_number(x) for x in uninterrupted):
            resume_pass = False

        if not all(is_finite_number(x) for x in resumed):
            resume_pass = False

    if not is_finite_number(tolerance) or float(tolerance) < 0:
        resume_pass = False

    if resume_pass:
        for a, b in zip(uninterrupted, resumed):
            if abs(float(a) - float(b)) > float(tolerance):
                resume_pass = False
                break

    if not resume_pass:
        reason_codes.add("RESUME_DIVERGENCE")

    # --------------------------------------------------------
    # Full model artifact check
    #
    # Only adapter artifacts are allowed. If extra model artifacts
    # are supplied, this is considered a full-model artifact.
    # --------------------------------------------------------

    if isinstance(artifact_files, list):
        if any(
            isinstance(x, str)
            and x not in expected_files
            for x in artifact_files
        ):
            reason_codes.add("FULL_MODEL_ARTIFACT")

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    reason_codes = utf8_sorted_unique(reason_codes)

    return {
        "labels": labels,
        "templatePass": body.get("templateApplications") == 1,
        "trainableParams": trainable_names,
        "trainableCount": trainable_count,
        "peftConfigPass": (
            params_valid
            and len(trainable) > 0
            and body.get("inferenceMode") is False
        ),
        "adapterFiles": sorted(
            adapter_files,
            key=lambda x: x.encode("utf-8")
        ),
        "checkpointComplete": checkpoint_pass,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_valid,
        "evaluationDeterministic": (
            body.get("dropoutActiveDuringEval") is False
        ),
        "resumePass": resume_pass,
        "reasonCodes": reason_codes,
    }


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/adapt")
async def adapt(request: Request):

    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    operation = body.get("operation")

    if operation not in {"choose", "repair"}:
        return invalid_input()

    if operation == "choose":
        return choose_operation(body)

    return repair_operation(body)


@app.get("/")
def root():
    return {"status": "ok"}

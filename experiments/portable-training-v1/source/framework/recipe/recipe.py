"""Project C's six-field recipe action. Authored source; not executed locally.

Only stdlib is imported here. The compiler returns source text; it never imports
or runs the native starter, changes its model, or changes the native step budget.
"""
import hashlib
import json
import math
import random


STARTER_SHA256 = "42b5161c57a4bed3ae7ebf9478e726542fcb7ba7d276f40fe4f9ef2bdce2cdd5"
RANGES = {
    "lr": (1e-5, 1e-2),
    "warmup_frac": (0.0, 0.3),
    "weight_decay": (0.0, 0.3),
    "grad_clip": (0.1, 10.0),
    "beta2": (0.9, 0.999),
}
SCHEDULES = ("cosine", "linear", "constant")
FIELDS = frozenset((*RANGES, "schedule"))


class RecipeError(ValueError):
    """An invalid candidate action; the caller may assign candidate-fault zero."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise RecipeError("duplicate recipe key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise RecipeError("nonfinite JSON number: " + value)


def _validated(recipe):
    if type(recipe) is not dict or set(recipe) != FIELDS:
        raise RecipeError("recipe must contain exactly the six declared fields")
    result = {}
    for name, (low, high) in RANGES.items():
        value = recipe[name]
        # Reject booleans and user-defined numeric objects before conversion.
        if type(value) not in (int, float):
            raise RecipeError(name + " must be a finite JSON number")
        try:
            valid = math.isfinite(value) and low <= value <= high
        except (OverflowError, ValueError):
            valid = False
        if not valid:
            raise RecipeError(name + " is nonfinite or outside its inclusive range")
        result[name] = float(value)
    schedule = recipe["schedule"]
    if type(schedule) is not str or schedule not in SCHEDULES:
        raise RecipeError("schedule must be cosine, linear or constant")
    result["schedule"] = schedule
    return result


def parse(text):
    """Parse one strict JSON object into a fresh validated six-field dict.

    Markdown fences, trailing objects/text, duplicates, missing/unknown fields,
    booleans, nonfinite numbers and out-of-range values are candidate errors.
    """
    if type(text) is not str:
        raise RecipeError("recipe action must be JSON text")
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except RecipeError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise RecipeError("invalid recipe JSON") from error
    return _validated(value)


def compile_recipe(recipe, starter_text):
    """Return complete pinned starter source with only the six recipe changes.

    No recipe text becomes Python syntax: numbers become validated float reprs
    and schedule chooses one of three fixed expressions. Progress is clamped to
    [0,1]. Positive warmup rises from zero to LR; decay starts at the warmup
    boundary and reaches zero for cosine/linear at progress1. Warmup0 starts at
    LR without division by zero. Constant holds LR after any warmup.
    """
    action = _validated(recipe)
    if type(starter_text) is not str:
        raise RecipeError("starter must be exact published UTF-8 text")
    try:
        digest = hashlib.sha256(starter_text.encode("utf-8")).hexdigest()
    except UnicodeError as error:
        raise RecipeError("starter must be exact published UTF-8 text") from error
    if digest != STARTER_SHA256:
        raise RecipeError("published starter SHA256 differs")
    decay = {
        "cosine": "LR * 0.5 * (1.0 + math.cos(math.pi * decay_progress))",
        "linear": "LR * (1.0 - decay_progress)",
        "constant": "LR",
    }[action["schedule"]]
    original_schedule = (
        "    lr = LR * frac / WARMUP_FRAC if frac < WARMUP_FRAC else \\\n"
        "        LR * 0.5 * (1 + math.cos(math.pi * (frac - WARMUP_FRAC) / (1 - WARMUP_FRAC)))\n"
    )
    schedule = (
        "    progress = min(1.0, max(0.0, float(frac)))\n"
        "    if WARMUP_FRAC > 0.0 and progress < WARMUP_FRAC:\n"
        "        lr = LR * progress / WARMUP_FRAC\n"
        "    else:\n"
        "        decay_progress = (progress - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)\n"
        "        lr = " + decay + "\n"
    )
    replacements = (
        ("LR, WARMUP_FRAC, WD = 1e-3, 0.03, 0.1\n",
         "LR, WARMUP_FRAC, WD = " + ", ".join(repr(action[key]) for key in
                                                ("lr", "warmup_frac", "weight_decay")) + "\n"),
        ("betas=(0.9, 0.95)", "betas=(0.9, " + repr(action["beta2"]) + ")"),
        (original_schedule, schedule),
        ("torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
         "torch.nn.utils.clip_grad_norm_(model.parameters(), " + repr(action["grad_clip"]) + ")"),
    )
    result = starter_text
    for old, new in replacements:
        if result.count(old) != 1:
            raise RecipeError("published starter replacement anchor differs")
        result = result.replace(old, new, 1)
    return result


def random_recipe(seed):
    """Fresh deterministic valid control action for an integer seed.

    Predeclared independent draws: log-uniform LR and clip; uniform warmup,
    decay and beta2; uniform choice from cosine/linear/constant. No global RNG
    state is read or changed. No score or model state influences sampling.
    """
    if type(seed) is not int:
        raise RecipeError("random recipe seed must be an integer")
    rng = random.Random(seed)

    def loguniform(name):
        low, high = RANGES[name]
        return min(high, max(low, math.exp(rng.uniform(math.log(low), math.log(high)))))

    return _validated({
        "lr": loguniform("lr"),
        "warmup_frac": rng.uniform(*RANGES["warmup_frac"]),
        "weight_decay": rng.uniform(*RANGES["weight_decay"]),
        "schedule": rng.choice(SCHEDULES),
        "grad_clip": loguniform("grad_clip"),
        "beta2": rng.uniform(*RANGES["beta2"]),
    })

"""Stub: Cron blueprints (removed)."""
CATALOG = {}
WEEKDAY_PRESETS = {}

def get_blueprint(*a, **kw): return None
def _humanize_schedule(*a, **kw): return "unavailable"

class BlueprintFillError(Exception): pass

def fill_blueprint(*a, **kw):
    raise BlueprintFillError("Cron blueprints removed")

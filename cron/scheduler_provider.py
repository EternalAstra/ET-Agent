"""Stub: Cron scheduler provider (removed)."""
def resolve_cron_scheduler(*a, **kw):
    from cron.scheduler import tick
    return type('Stub', (), {'tick': tick, 'start': lambda s: None})()

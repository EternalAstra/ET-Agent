"""Stub: Cron jobs (removed in trimmed version)."""
import threading

_jobs_lock = threading.Lock()

def load_jobs(): return {}
def save_jobs(*a, **kw): pass
def list_jobs(*a, **kw): return []
def create_job(*a, **kw): raise NotImplementedError("Cron removed")
def get_job(*a, **kw): return None
def update_job(*a, **kw): raise NotImplementedError("Cron removed")
def remove_job(*a, **kw): raise NotImplementedError("Cron removed")
def pause_job(*a, **kw): raise NotImplementedError("Cron removed")
def resume_job(*a, **kw): raise NotImplementedError("Cron removed")
def trigger_job(*a, **kw): raise NotImplementedError("Cron removed")
def parse_schedule(*a, **kw): raise ValueError("Cron removed")
def rewrite_skill_refs(*a, **kw): pass

class AmbiguousJobReference(Exception): pass

def resolve_job_ref(*a, **kw):
    raise AmbiguousJobReference("Cron removed")

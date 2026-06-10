"""The uvicorn general server logger is named ``uvicorn.error`` even for routine
INFO lifecycle lines. `_UvicornRelabelFilter` relabels it to ``uvicorn`` so the
logger name stops implying "error"; the levelname remains the source of truth for
severity. These tests pin that behaviour (and that nothing else is touched)."""
import logging

from app.main import _UvicornRelabelFilter


def _rec(name: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg="x", args=(), exc_info=None,
    )


def test_relabels_uvicorn_error_to_uvicorn():
    f = _UvicornRelabelFilter()
    rec = _rec("uvicorn.error")
    assert f.filter(rec) is True          # never drops the record
    assert rec.name == "uvicorn"          # misleading label corrected


def test_relabels_regardless_of_level():
    """A genuine uvicorn error (ERROR level) is still relabeled to 'uvicorn' —
    severity is preserved via levelname, so nothing is hidden."""
    f = _UvicornRelabelFilter()
    rec = _rec("uvicorn.error", level=logging.ERROR)
    assert f.filter(rec) is True
    assert rec.name == "uvicorn"
    assert rec.levelno == logging.ERROR


def test_leaves_other_loggers_untouched():
    f = _UvicornRelabelFilter()
    for name in ("uvicorn", "uvicorn.access", "app.news.sources.fmp_source"):
        rec = _rec(name)
        assert f.filter(rec) is True
        assert rec.name == name

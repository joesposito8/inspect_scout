"""Materialize finding answers after scans."""

import logging

from inspect_scout._project._project import read_project
from inspect_scout._recorder.recorder import Status
from inspect_scout._scanlist import scan_list_async

from .answer import refresh_answer
from .derive import _scanner_findings
from .spec import list_findings

logger = logging.getLogger(__name__)


async def finalize_findings(status: Status) -> None:
    """Refresh the answers of registered findings whose scanners ran in this scan."""
    # A scan that ran to its end is complete or has errors (those transcripts
    # count as partial coverage); one with neither was interrupted and is
    # refreshed when it is resumed.
    if not status.complete and not status.errors:
        return
    slugs = {
        slug
        for name, scanner in status.spec.scanners.items()
        for slug in _scanner_findings(name, scanner)
    }
    if not slugs:
        return
    selected = [spec for spec in list_findings() if spec.slug in slugs]
    if not selected:
        return
    project = read_project()
    scans = project.scans or "./scans"
    if status.spec.scan_id not in {
        s.spec.scan_id for s in await scan_list_async(scans)
    }:
        logger.warning(
            "Scan %s is outside the project's scans directory (%s), so its finding "
            "results are not part of any finding.",
            status.spec.scan_id,
            scans,
        )
        return
    for spec in selected:
        # The scan's results are already saved; a failure here must not discard
        # them or skip the remaining findings.
        try:
            await refresh_answer(spec, project)
        except Exception as ex:
            logger.warning(
                "The answer for finding %s was not updated after scan %s: %s. "
                "Regenerate it with `scout findings answer %s`.",
                spec.slug,
                status.spec.scan_id,
                ex,
                spec.slug,
            )

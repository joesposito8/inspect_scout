"""Commands for project findings."""

import getpass
from collections import Counter

import click
from inspect_ai._util._async import run_coroutine

from inspect_scout._findings.answer import answer_headline, refresh_answer
from inspect_scout._findings.derive import Exhibit, FindingView, derive_finding
from inspect_scout._findings.judge import (
    JudgeScope,
    decide_exhibits,
    record_judge_run,
    reviewed_examples,
    select_candidates,
)
from inspect_scout._findings.links import view_url
from inspect_scout._findings.spec import list_findings, read_finding, register_finding
from inspect_scout._findings.verdicts import (
    VerdictRow,
    VerdictState,
    append_verdict,
    retired_verdicts_warning,
)
from inspect_scout._project._project import read_project
from inspect_scout._recorder.recorder import Status


@click.group(name="findings")
def findings_command() -> None:
    """Register standing questions and inspect their evidence."""


@findings_command.command(name="register")
@click.argument("slug")
@click.option("--prompt", required=True, help="Question for the finding scanner.")
@click.option("--title", default=None, help="Display title.")
@click.option("--label", multiple=True, help="Allowed label (repeat for each label).")
def register_command(
    slug: str, prompt: str, title: str | None, label: tuple[str, ...]
) -> None:
    """Create or update a finding in this project."""
    try:
        spec = register_finding(slug, prompt, title=title, labels=list(label) or None)
    except (ValueError, OSError) as ex:
        raise click.ClickException(str(ex)) from ex
    click.echo(f"{spec.slug}: {spec.title} ({spec.prompt_sha})")


@findings_command.command(name="list")
def list_command() -> None:
    """List registered findings."""
    for spec in list_findings():
        click.echo(f"{spec.slug}\t{spec.title}\t{spec.prompt_sha}")


@findings_command.command(name="judge")
@click.argument("slug")
@click.option(
    "--only",
    type=click.Choice(["proposed", "all"]),
    default="proposed",
    show_default=True,
    help="proposed: exhibits nobody has decided; all: also re-decide the judge's earlier decisions. Human decisions are never re-judged.",
)
@click.option(
    "--model", default=None, help="Judge model (default: the findings model)."
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Judge at most N exhibits.",
)
def judge_command(
    slug: str, only: JudgeScope, model: str | None, limit: int | None
) -> None:
    """Confirm or reject pending exhibits, learning from the human decisions."""
    try:
        spec = read_finding(slug)
    except (ValueError, OSError) as ex:
        raise click.ClickException(str(ex)) from ex
    if warning := retired_verdicts_warning(spec):
        click.echo(warning, err=True)
    project = read_project()
    view = derive_finding(spec, project)
    candidates, unlocated = select_candidates(view, only, limit)
    click.echo(
        f"{slug}: judging {len(candidates)} exhibits with {model or project.findings.model} ({unlocated} skipped: unlocated)"
    )
    outcome = run_coroutine(
        decide_exhibits(spec, project, candidates, reviewed_examples(view), model=model)
    )
    run = record_judge_run(spec, outcome, only)
    run_coroutine(refresh_answer(spec, project))
    cost = f"${run.cost:.4f}" if run.cost is not None else "price unknown"
    click.echo(
        f"{run.confirmed} confirmed, {run.rejected} rejected, {run.refused} refused by judge, {run.failed} failed; "
        f"{run.input_tokens} input / {run.input_tokens_cache_write} cache-write / {run.input_tokens_cache_read} cache-read / {run.output_tokens} output tokens, {cost}"
    )
    click.echo(derive_finding(spec, project).judge_agreement.line)


@findings_command.command(name="answer")
@click.argument("slug")
@click.option("--model", default=None)
@click.option("--no-synthesis", is_flag=True)
def answer_command(slug: str, model: str | None, no_synthesis: bool) -> None:
    """Explicitly regenerate the current answer, optionally without model synthesis."""
    path = run_coroutine(
        refresh_answer(
            read_finding(slug),
            read_project(),
            model=model,
            no_synthesis=no_synthesis,
            force=True,
        )
    )
    click.echo(str(path))


@findings_command.command(name="status")
def status_command() -> None:
    """Summarize evidence, coverage, review agreement and the current answer."""
    for spec in list_findings():
        if warning := retired_verdicts_warning(spec):
            click.echo(warning, err=True)
        view = derive_finding(spec)
        states = Counter(e.state for e in view.exhibits)
        coverage = Counter(view.coverage.get(spec.prompt_sha, {}).values())
        state_text = ", ".join(
            f"{states[state]} {state}"
            for state in ("proposed", "confirmed", "rejected")
        )
        agreement = view.agreements[-1].line if view.agreements else "no agreement yet"
        refused = (
            f"; {view.last_judge_run.refused} refused by judge"
            if view.last_judge_run
            else ""
        )
        click.echo(
            f"{spec.slug}: {state_text}{refused}; {sum(len(e.sightings) for e in view.exhibits)} sightings; coverage {coverage['judged']} judged / {coverage['partial']} partial / {coverage['missing']} missing; {agreement}; {view.judge_agreement.line}; {answer_headline(spec.slug)}"
        )


@findings_command.command(name="rotate-nonce")
def rotate_nonce_command() -> None:
    """Replace the project rendering nonce to invalidate scanner generation caches."""
    from shortuuid import uuid

    from inspect_scout._project._project import write_project_config
    from inspect_scout._project.types import ProjectConfig

    write_project_config(ProjectConfig(nonce=uuid()))
    click.echo("Project nonce rotated; future scans use a new rendering cache key.")


@findings_command.command(name="show")
@click.argument("slug")
@click.option(
    "--json", "as_json", is_flag=True, help="Print the complete derived view as JSON."
)
def show_command(slug: str, as_json: bool) -> None:
    """Show quote-verified exhibits and coverage across project scans."""
    try:
        spec = read_finding(slug)
        view = derive_finding(spec)
    except (ValueError, OSError) as ex:
        raise click.ClickException(str(ex)) from ex
    if as_json:
        click.echo(view.model_dump_json(indent=2))
        return
    click.echo(f"{spec.title}\n{spec.prompt}")
    print_finding_summary(view)
    for exhibit in view.exhibits:
        print_exhibit(exhibit)


def print_exhibit(exhibit: Exhibit) -> None:
    """Print the evidence and its independent scanner and decision provenance."""
    proposers = (
        ", ".join(
            sorted(
                {f"{s.scanner} ({s.model or 'no model'})" for s in exhibit.sightings}
            )
        )
        or "manual"
    )
    click.echo(
        f"\n{exhibit.task_id or exhibit.transcript_id} | "
        f"[{exhibit.message_label or '?'}] | #{exhibit.message_index or '?'} | "
        f"{exhibit.verification} | {exhibit.message_id}\n"
        f"{exhibit.state} | {exhibit.label or 'unlabeled'} | {len(exhibit.sightings)} sightings in "
        f"{exhibit.corroboration} scans\n{exhibit.quote}\n{exhibit.reason}\n"
        f"Proposed by: {proposers}"
    )
    if exhibit.decided_by:
        click.echo(f"Decision: {exhibit.decided_by}: {exhibit.verdict_reason}")


@findings_command.command(name="review")
@click.argument("slug")
@click.option(
    "--all", "review_all", is_flag=True, help="Include previously reviewed exhibits."
)
def review_command(slug: str, review_all: bool) -> None:
    """Review evidence: c confirm, r reject, l LABEL, s skip, o open, q quit."""
    spec = read_finding(slug)
    project = read_project()
    view = derive_finding(spec, project)
    try:
        _review_exhibits(view, project.transcripts, review_all)
    finally:
        run_coroutine(refresh_answer(spec, project))


def _review_exhibits(
    view: FindingView, transcripts: str | None, review_all: bool
) -> None:
    spec = view.finding
    actions: dict[str, VerdictState] = {
        "c": "confirmed",
        "r": "rejected",
        "l": "confirmed",
    }
    for exhibit in view.exhibits:
        if (
            not review_all
            and exhibit.state != "proposed"
            and exhibit.verification != "unlocated"
        ):
            continue
        print_exhibit(exhibit)
        location = exhibit.transcripts_location or transcripts
        link = (
            view_url(location, exhibit.transcript_id, exhibit.verified_ref)
            if location and exhibit.verified_ref
            else None
        )
        click.echo(link or "No located message link")
        # An unlocated quote matches no message, so there is nothing to confirm.
        allowed: dict[str, VerdictState] = (
            actions if exhibit.verified_ref else {"r": "rejected"}
        )
        keys = "c/r/l LABEL/s/o/q" if exhibit.verified_ref else "r/s/q"
        while True:
            command = click.prompt(keys, default="s").strip()
            action, _, label = command.partition(" ")
            if action == "q":
                return
            if action == "s":
                break
            if action == "o":
                if link:
                    click.launch(link)
                continue
            if action not in allowed:
                click.echo(f"Use {keys.replace('/', ', ')}.")
                continue
            if action == "l" and label not in (spec.labels or []):
                click.echo(f"Choose a label from: {', '.join(spec.labels or [])}")
                continue
            selected_label = (
                None
                if action == "r"
                else label
                if action == "l"
                else exhibit.label
                if exhibit.label in (spec.labels or [])
                else None
            )
            append_verdict(
                spec,
                VerdictRow(
                    message_id=exhibit.message_id,
                    verdict=actions[action],
                    label=selected_label,
                    decided_by=f"human:{getpass.getuser()}",
                    reason=f"{actions[action].capitalize()} during review",
                    source="manual",
                ),
            )
            click.echo(f"{exhibit.message_id}: {actions[action]}")
            break


def print_finding_summary(view: FindingView) -> None:
    """Print exhibit and current-prompt coverage counts."""
    coverage = view.coverage.get(view.finding.prompt_sha, {})
    counts = {
        state: list(coverage.values()).count(state)
        for state in ("judged", "partial", "missing")
    }
    relocated = sum(exhibit.verification == "relocated" for exhibit in view.exhibits)
    unlocated = sum(exhibit.verification == "unlocated" for exhibit in view.exhibits)
    click.echo(
        f"{view.finding.slug}: {len(view.exhibits)} exhibits, "
        f"{sum(len(exhibit.sightings) for exhibit in view.exhibits)} sightings; "
        f"{relocated} relocated, {unlocated} unlocated; "
        f"coverage {counts['judged']} judged, {counts['partial']} partial, {counts['missing']} missing"
    )
    if view.agreements:
        click.echo(
            f"Agreement ({view.agreements[-1].scan_id}): {view.agreements[-1].line}"
        )


def print_scan_findings(status: Status) -> None:
    """Print finding summaries after resuming or explicitly completing a scan."""
    from inspect_scout._findings.derive import _scanner_findings

    slugs = {
        slug
        for name, scanner in status.spec.scanners.items()
        for slug in _scanner_findings(name, scanner)
    }
    for spec in list_findings():
        if spec.slug in slugs:
            print_finding_summary(derive_finding(spec))

"""Metrics table formatting.

Formats sweep results as markdown or LaTeX tables.
No model-specific knowledge.
"""


def format_table(
    rows: list,
    columns: list,
    fmt: str = 'markdown',
    float_fmt: str = '.4f',
    caption: str | None = None,
) -> str:
    """Format rows as a table string.

    Parameters
    ----------
    rows : list of flat dicts (e.g. from sweep()).
        Each dict maps column names to values.
    columns : ordered column names to include.
    fmt : 'markdown' or 'latex'.
    float_fmt : format spec for floats.
    caption : LaTeX caption (ignored for markdown).
    """
    if fmt == 'latex':
        return _format_latex(rows, columns, float_fmt, caption)
    return _format_markdown(rows, columns, float_fmt)


def _format_cell(val, float_fmt):
    """Format a single cell value."""
    if isinstance(val, float):
        return format(val, float_fmt)
    return str(val)


def _format_markdown(rows, columns, float_fmt):
    """Render rows as a markdown table."""
    # Compute column widths
    widths = {c: len(c) for c in columns}
    formatted = []
    for row in rows:
        cells = {}
        for c in columns:
            s = _format_cell(row.get(c, ''), float_fmt)
            cells[c] = s
            widths[c] = max(widths[c], len(s))
        formatted.append(cells)

    lines = []
    # Header
    header = '| ' + ' | '.join(c.ljust(widths[c]) for c in columns) + ' |'
    sep = '|' + '|'.join('-' * (widths[c] + 2) for c in columns) + '|'
    lines.append(header)
    lines.append(sep)

    # Rows
    for cells in formatted:
        line = '| ' + ' | '.join(
            cells[c].rjust(widths[c]) for c in columns) + ' |'
        lines.append(line)

    return '\n'.join(lines)


def _format_latex(rows, columns, float_fmt, caption):
    """Render rows as a LaTeX tabular."""
    col_spec = 'l' * len(columns)
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Header
    header = ' & '.join(_latex_escape(c) for c in columns) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    # Rows
    for row in rows:
        cells = [_format_cell(row.get(c, ''), float_fmt) for c in columns]
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    if caption:
        lines.append(r'\caption{' + _latex_escape(caption) + '}')

    lines.append(r'\end{table}')
    return '\n'.join(lines)


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters in text."""
    for char in ('_', '%', '&', '#', '$'):
        s = s.replace(char, '\\' + char)
    return s


def write_table(
    path: str,
    rows: list,
    columns: list,
    fmt: str = 'markdown',
    **kwargs,
) -> None:
    """Format and write table to file."""
    from pathlib import Path as _Path
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = format_table(rows, columns, fmt=fmt, **kwargs)
    p.write_text(content)

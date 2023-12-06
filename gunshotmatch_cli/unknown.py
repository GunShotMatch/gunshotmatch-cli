#!/usr/bin/env python3
#
#  decision_tree.py
"""
Pipeline for unknown propellant/OGSR sample.
"""
#
#  Copyright © 2020-2023 Dominic Davis-Foster <dominic@davis-foster.co.uk>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
#

# 3rd party
import click

__all__ = ("decision_tree", )


@click.argument("unknown_toml", default="unknown.toml")
@click.command()
def unknown(unknown_toml: str = "unknown.toml") -> None:
	"""
	Pipeline for unknown propellant/OGSR sample.
	"""

	# 3rd party
	from domdf_python_tools.paths import PathPlus
	from gunshotmatch_pipeline.exporters import write_matches_json
	from gunshotmatch_pipeline.unknowns import UnknownSettings, process_unknown

	unknown = UnknownSettings.from_toml(PathPlus(unknown_toml).read_text())
	# print(unknown)
	# print(unknown.load_method())
	# print(unknown.load_config())
	print("Processing unknown", unknown.name)

	project = process_unknown(unknown, unknown.output_directory, recreate=False)

	write_matches_json(project, PathPlus(unknown.output_directory))
	assert project.consolidated_peaks is not None
	print(len(project.consolidated_peaks))

	# print(ms_comparison_df)


if __name__ == "__main__":
	unknown()

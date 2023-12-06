#!/usr/bin/env python3
#
#  versions.py
"""
Tool to get software versions.
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

# stdlib
import importlib.metadata
import platform
import sys
import textwrap

# 3rd party
import click
from domdf_python_tools.stringlist import StringList
from domdf_python_tools.words import LF

# this package
import gunshotmatch_cli

__all__ = ("get_formatted_versions", "version_callback")


def get_formatted_versions() -> StringList:
	"""
	Return the versions of this software and its dependencies, one per line.
	"""

	message = StringList()

	our_version = gunshotmatch_cli.__version__
	message.append(f"Version: {our_version}")

	pipeline_version = importlib.metadata.version("gunshotmatch-pipeline")
	message.append(f"GunShotMatch Pipeline: {pipeline_version}")

	libgsm_version = importlib.metadata.version("libgunshotmatch")
	message.append(f"LibGunShotMatch: {libgsm_version}")

	mpl_version = importlib.metadata.version("scikit-learn")
	message.append(f"scikit-learn: {mpl_version}")

	message.append(f"Python: {sys.version.replace(LF, ' ')}")

	message.append(' '.join(platform.system_alias(platform.system(), platform.release(), platform.version())))

	return message


def version_callback(ctx: click.Context, param: click.Option, value: int) -> None:
	"""
	Callback for displaying the package version (and optionally the Python runtime).
	"""

	# this package
	import gunshotmatch_cli

	if not value or ctx.resilient_parsing:
		return

	if value > 2:
		versions = textwrap.indent(
				str(get_formatted_versions()),
				"  ",
				)
		click.echo("GunShotMatch CLI")
		click.echo(versions)
	elif value > 1:
		python_version = sys.version.replace('\n', ' ')
		click.echo(f"GunShotMatch CLI version {gunshotmatch_cli.__version__}, Python {python_version}")
	else:
		click.echo(f"GunShotMatch CLI version {gunshotmatch_cli.__version__}")

	ctx.exit()

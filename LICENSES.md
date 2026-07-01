# Licensing

This repository bundles multiple upstream projects. The overall
redistribution story depends on all included upstream licenses.

## Wrapper / orchestration scripts

- **Location:** `scripts/`, `bridge/`, `Makefile`
- **License:** MIT, unless otherwise stated.
- **Copyright:** 2026 dawsonblock

These are the original wrapper scripts that orchestrate the dubbing
pipeline. They are not derived from any upstream project.

## pyVideoTrans

- **Location:** `pyvideotrans-main/`
- **License:** GPLv3
- **Upstream:** https://github.com/jianchang512/pyvideotrans

pyVideoTrans is a full-featured video translation and dubbing tool. It is
licensed under the GNU General Public License v3, which means any
redistribution of the full bundle (wrapper + pyVideoTrans) must comply with
GPLv3 terms.

## Redistribution notice

Redistribution of the full bundle (wrapper scripts + pyVideoTrans) is
subject to **all** included upstream licenses. Because pyVideoTrans is
GPLv3, the combined work must be distributed under GPLv3 terms. The
MIT-licensed wrapper scripts can still be used independently under MIT
terms, but when distributed together with pyVideoTrans, the GPLv3 license
takes precedence for the combined work.

**In short:** do not assume the whole project is MIT. It is not.

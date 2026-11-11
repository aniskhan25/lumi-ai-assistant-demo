#!/bin/bash
# Print the first traceback (with its exception line) of the newest tiny-serve log, or of $1.
F="${1:-$(ls -t "$(dirname "$0")"/../../benchmarks/results/dsv41-tiny/*.log | head -n1)}"
grep -vE "amdsmi|pynvml|FutureWarning" "$F" | sed -E 's/^\((EngineCore|APIServer|Worker[_A-Z0-9]*) pid=[0-9]+\) (ERROR [0-9-]+ [0-9:]+ \[[^]]+\] )?//' \
  | awk '/Traceback \(most recent call last\)/{c++} c==1{print; if (/^[A-Za-z_.]*(Error|Exception)[:]?/ && !/^  /) exit}' | grep -vE '^\s+\^+$' | tail -${N:-40}

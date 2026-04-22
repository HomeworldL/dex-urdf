#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDS_DIR="${ROOT_DIR}/robots/hands"
MIGRATE="${ROOT_DIR}/tools/migrate_hand_urdf_to_mjcf.py"
LOG_DIR="${ROOT_DIR}/robots_mjcf/hands/_logs"
SUMMARY="${LOG_DIR}/migrate_all_summary.txt"

mkdir -p "${LOG_DIR}"
: > "${SUMMARY}"

append_unique() {
  local value="$1"
  local existing
  for existing in "${SOURCES[@]}"; do
    if [[ "${existing}" == "${value}" ]]; then
      return 0
    fi
  done
  SOURCES+=("${value}")
}

infer_side() {
  local source_path="$1"
  local base
  base="$(basename "${source_path}" .urdf)"
  if [[ "${base}" == *_left* ]]; then
    echo "left"
  elif [[ "${base}" == *_right* ]]; then
    echo "right"
  else
    echo "right"
  fi
}

echo "[info] writing logs to ${LOG_DIR}"

while IFS= read -r -d '' hand_dir; do
  hand_name="${hand_dir#${HANDS_DIR}/}"
  safe_hand_name="${hand_name//\//__}"
  SOURCES=()

  while IFS= read -r -d '' urdf; do
    append_unique "${urdf}"
  done < <(
    find "${hand_dir}" -maxdepth 1 \
      -name "*_glb.urdf" \
      -print0 | sort -z
  )

  if [[ "${#SOURCES[@]}" -eq 0 ]]; then
    printf '[skip] %s: no *_glb.urdf sources found\n' "${hand_name}" | tee -a "${SUMMARY}"
    continue
  fi

  for source_urdf in "${SOURCES[@]}"; do
    rel_source="${source_urdf#${ROOT_DIR}/}"
    log_name="${safe_hand_name}__$(basename "${source_urdf}" .urdf).log"
    log_path="${LOG_DIR}/${log_name}"
    side="$(infer_side "${source_urdf}")"

    printf '[run] %s <- %s\n' "${hand_name}" "${rel_source}" | tee -a "${SUMMARY}"
    if python "${MIGRATE}" --hand-name "${hand_name}" --side "${side}" >"${log_path}" 2>&1 &&
      ! grep -q '^\[warn\]' "${log_path}"; then
      printf '[ok]  %s <- %s\n' "${hand_name}" "${rel_source}" | tee -a "${SUMMARY}"
    else
      printf '[fail] %s <- %s (see %s)\n' "${hand_name}" "${rel_source}" "${log_path}" | tee -a "${SUMMARY}"
    fi
  done
done < <(
  find "${HANDS_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 |
    sort -z
)

echo "[done] summary: ${SUMMARY}"

#!/usr/bin/env bash
# ==============================================================================
# organize_local_changes.sh
#
# Organiza modificações locais do repositório carga-termica-bovinos sem usar
# "git add .".
#
# O script faz:
#   1. Confere se está dentro de um repositório Git.
#   2. Cria uma branch de trabalho, se necessário.
#   3. Gera backup do diff e da lista de arquivos não rastreados.
#   4. Atualiza/cria .gitignore com regras para outputs, profiling e dados.
#   5. Cria __init__.py em diretórios essenciais.
#   6. Remove do índice arquivos gerados, caso tenham sido rastreados.
#   7. Adiciona seletivamente apenas código, configuração e documentação.
#   8. Executa checagens básicas.
#   9. Opcionalmente cria o commit.
#
# Uso recomendado:
#
#   bash scripts/organize_local_changes.sh --dry-run
#   bash scripts/organize_local_changes.sh --apply
#   bash scripts/organize_local_changes.sh --apply --commit
#
# ==============================================================================

set -Eeuo pipefail

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
DRY_RUN=1
DO_COMMIT=0
RUN_CHECKS=1
BRANCH_NAME="refactor/thermal-comfort-pipeline"
COMMIT_MESSAGE="Refactor thermal comfort pipeline and organize generated outputs"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
info() {
    printf '[INFO] %s\n' "$*"
}

warn() {
    printf '[WARN] %s\n' "$*" >&2
}

error() {
    printf '[ERROR] %s\n' "$*" >&2
}

die() {
    error "$*"
    exit 1
}

run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[DRY-RUN] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
show_help() {
    cat <<EOF
Uso:
  bash organize_local_changes.sh [opções]

Opções:
  --dry-run              Mostra o que seria feito, sem modificar nada. Padrão.
  --apply                Executa de fato as alterações.
  --commit               Cria commit após organizar e fazer git add seletivo.
  --no-checks            Não roda checagens Python/Git ao final.
  --branch NAME          Nome da branch de trabalho.
                         Padrão: ${BRANCH_NAME}
  --message MESSAGE      Mensagem do commit.
                         Padrão: ${COMMIT_MESSAGE}
  -h, --help             Mostra esta ajuda.

Exemplos:
  bash organize_local_changes.sh --dry-run
  bash organize_local_changes.sh --apply
  bash organize_local_changes.sh --apply --commit
EOF
}

# ------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --apply)
            DRY_RUN=0
            shift
            ;;
        --commit)
            DO_COMMIT=1
            shift
            ;;
        --no-checks)
            RUN_CHECKS=0
            shift
            ;;
        --branch)
            [[ $# -ge 2 ]] || die "A opção --branch exige um argumento."
            BRANCH_NAME="$2"
            shift 2
            ;;
        --message)
            [[ $# -ge 2 ]] || die "A opção --message exige um argumento."
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            die "Opção desconhecida: $1"
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Preconditions
# ------------------------------------------------------------------------------
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Este diretório não é um repositório Git."

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CURRENT_BRANCH="$(git branch --show-current || true)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR=".git_safety_backup/${TIMESTAMP}"

info "Repositório: ${REPO_ROOT}"
info "Branch atual: ${CURRENT_BRANCH:-detached}"
info "Branch alvo: ${BRANCH_NAME}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    warn "Modo dry-run ativo. Nada será alterado."
else
    warn "Modo apply ativo. O script fará alterações locais."
fi

# ------------------------------------------------------------------------------
# Safety backup
# ------------------------------------------------------------------------------
info "Criando backup de segurança do estado local."

run mkdir -p "${BACKUP_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[DRY-RUN] git diff > %q\n' "${BACKUP_DIR}/working_tree.diff"
    printf '[DRY-RUN] git diff --staged > %q\n' "${BACKUP_DIR}/staged.diff"
    printf '[DRY-RUN] git status --short > %q\n' "${BACKUP_DIR}/status_short.txt"
    printf '[DRY-RUN] git ls-files --others --exclude-standard > %q\n' "${BACKUP_DIR}/untracked_files.txt"
else
    git diff > "${BACKUP_DIR}/working_tree.diff" || true
    git diff --staged > "${BACKUP_DIR}/staged.diff" || true
    git status --short > "${BACKUP_DIR}/status_short.txt" || true
    git ls-files --others --exclude-standard > "${BACKUP_DIR}/untracked_files.txt" || true
fi

# ------------------------------------------------------------------------------
# Branch
# ------------------------------------------------------------------------------
if [[ "${CURRENT_BRANCH}" != "${BRANCH_NAME}" ]]; then
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        info "Branch ${BRANCH_NAME} já existe. Trocando para ela."
        run git switch "${BRANCH_NAME}"
    else
        info "Criando branch ${BRANCH_NAME}."
        run git switch -c "${BRANCH_NAME}"
    fi
else
    info "Já está na branch ${BRANCH_NAME}."
fi

# ------------------------------------------------------------------------------
# Ensure package markers
# ------------------------------------------------------------------------------
info "Garantindo __init__.py em diretórios de pacote."

PACKAGE_DIRS=(
    "app"
    "app/pipeline"
    "app/plot"
    "app/io"
    "app/time"
    "app/util"
    "app/pipeline/thermal_comfort"
)

for dir in "${PACKAGE_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        run touch "${dir}/__init__.py"
    fi
done

# ------------------------------------------------------------------------------
# .gitignore
# ------------------------------------------------------------------------------
info "Atualizando .gitignore com bloco controlado."

GITIGNORE_BLOCK_BEGIN="# >>> carga-termica-bovinos generated outputs >>>"
GITIGNORE_BLOCK_END="# <<< carga-termica-bovinos generated outputs <<<"

read -r -d '' GITIGNORE_BLOCK <<'EOF' || true
# >>> carga-termica-bovinos generated outputs >>>

# Generated outputs
outputs_conforto/
app/outputs_conforto/
app/figures_article/
app/resultados_dissertacao/
figures/

# Profiling
*.prof
profile.prof

# Generated scientific figures and tables
*.png
*.pdf

# Local/generated datasets
*.parquet
*.csv
*.xlsx

# Python cache/environment
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
.env/

# Local editor/OS files
.DS_Store
Thumbs.db

# Safety backups created by this script
.git_safety_backup/

# <<< carga-termica-bovinos generated outputs <<<
EOF

if [[ ! -f ".gitignore" ]]; then
    run touch ".gitignore"
fi

if grep -qF "${GITIGNORE_BLOCK_BEGIN}" ".gitignore" 2>/dev/null; then
    info ".gitignore já possui o bloco controlado. Substituindo bloco antigo."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[DRY-RUN] replace controlled block in .gitignore\n'
    else
        python - <<'PY'
from pathlib import Path

path = Path(".gitignore")
text = path.read_text(encoding="utf-8")

begin = "# >>> carga-termica-bovinos generated outputs >>>"
end = "# <<< carga-termica-bovinos generated outputs <<<"

block = """# >>> carga-termica-bovinos generated outputs >>>

# Generated outputs
outputs_conforto/
app/outputs_conforto/
app/figures_article/
app/resultados_dissertacao/
figures/

# Profiling
*.prof
profile.prof

# Generated scientific figures and tables
*.png
*.pdf

# Local/generated datasets
*.parquet
*.csv
*.xlsx

# Python cache/environment
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
.env/

# Local editor/OS files
.DS_Store
Thumbs.db

# Safety backups created by this script
.git_safety_backup/

# <<< carga-termica-bovinos generated outputs <<<
"""

start = text.index(begin)
stop = text.index(end) + len(end)
new_text = text[:start].rstrip() + "\n\n" + block.rstrip() + "\n" + text[stop:].lstrip()
path.write_text(new_text, encoding="utf-8")
PY
    fi
else
    info "Adicionando bloco controlado ao .gitignore."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[DRY-RUN] append generated outputs block to .gitignore\n'
    else
        {
            printf '\n'
            printf '%s\n' "${GITIGNORE_BLOCK}"
        } >> ".gitignore"
    fi
fi

# ------------------------------------------------------------------------------
# Unstage everything
# ------------------------------------------------------------------------------
info "Limpando área staged antes do add seletivo."
run git restore --staged :/

# ------------------------------------------------------------------------------
# Remove generated tracked files from index if they were accidentally tracked
# ------------------------------------------------------------------------------
info "Removendo do índice arquivos gerados, sem apagar do disco."

GENERATED_PATHS=(
    "outputs_conforto"
    "app/outputs_conforto"
    "app/figures_article"
    "app/resultados_dissertacao"
    "figures"
    "profile.prof"
)

for path in "${GENERATED_PATHS[@]}"; do
    if git ls-files --error-unmatch "${path}" >/dev/null 2>&1; then
        run git rm -r --cached --ignore-unmatch "${path}"
    else
        # If the path is a directory, git ls-files --error-unmatch path may not match.
        if [[ -n "$(git ls-files "${path}" 2>/dev/null)" ]]; then
            run git rm -r --cached --ignore-unmatch "${path}"
        fi
    fi
done

# ------------------------------------------------------------------------------
# Selective add
# ------------------------------------------------------------------------------
info "Adicionando seletivamente código, configuração e documentação."

add_if_exists() {
    local path="$1"
    if [[ -e "${path}" || -L "${path}" ]]; then
        run git add "${path}"
    else
        warn "Ignorando caminho ausente: ${path}"
    fi
}

CODE_PATHS=(
    ".gitignore"
    "README.md"
    "LICENSE"
    "requirements.txt"
    "pyproject.toml"

    "app/__init__.py"
    "app/chart_config.yaml"
    "app/config.py"
    "app/config_schema.py"
    "app/config.yaml"
    "app/extract_comfort_periods.py"
    "app/run.py"
    "app/run_pipeline.py"
    "app/thermal_comfort_pipeline.py"
    "app/biothermal_efficiency_index.py"
    "app/Thermal_Saturation_Efficiency_Model.py"
    "app/generate_article_figures.py"

    "app/io"
    "app/time"
    "app/util"
    "app/pipeline/__init__.py"
    "app/pipeline/density.py"
    "app/pipeline/geometry.py"
    "app/pipeline/smoothing.py"
    "app/pipeline/zones.py"
    "app/pipeline/thermal_comfort"
    "app/plot/__init__.py"
    "app/plot/plot_psychro.py"

    "psychrometrics_ashrae_si.py"
)

for path in "${CODE_PATHS[@]}"; do
    add_if_exists "${path}"
done

# ------------------------------------------------------------------------------
# Show what is staged
# ------------------------------------------------------------------------------
info "Resumo do que ficou staged."
run git diff --cached --stat

if [[ "${DRY_RUN}" -eq 0 ]]; then
    echo
    git status --short
    echo
fi

# ------------------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------------------
if [[ "${RUN_CHECKS}" -eq 1 ]]; then
    info "Rodando checagens básicas."

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[DRY-RUN] python -m compileall -q app psychrometrics_ashrae_si.py\n'
    else
        if command -v python >/dev/null 2>&1; then
            python -m compileall -q app psychrometrics_ashrae_si.py || {
                warn "compileall encontrou erro. Corrija antes do commit definitivo."
            }
        else
            warn "Python não encontrado no PATH. Pulando compileall."
        fi
    fi
fi

# ------------------------------------------------------------------------------
# Commit
# ------------------------------------------------------------------------------
if [[ "${DO_COMMIT}" -eq 1 ]]; then
    info "Criando commit."

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[DRY-RUN] git commit -m %q\n' "${COMMIT_MESSAGE}"
    else
        if git diff --cached --quiet; then
            warn "Nada staged para commit. Commit não criado."
        else
            git commit -m "${COMMIT_MESSAGE}"
        fi
    fi
else
    info "Commit não solicitado. Revise com:"
    echo
    echo "  git status"
    echo "  git diff --cached --stat"
    echo "  git diff --cached"
    echo
    echo "Para commitar depois:"
    echo
    echo "  git commit -m \"${COMMIT_MESSAGE}\""
fi

# ------------------------------------------------------------------------------
# Final instructions
# ------------------------------------------------------------------------------
info "Organização concluída."

cat <<EOF

Próximos comandos úteis:

  git status
  git diff --cached --stat
  git diff --cached

Para testar execução como pacote:

  python -m app.run --config app/config.yaml

Para subir a branch:

  git push -u origin ${BRANCH_NAME}

Backup de segurança criado em:

  ${BACKUP_DIR}

EOF

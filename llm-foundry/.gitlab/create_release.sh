#!/bin/bash

# Функция для API вызовов
create_gitlab_entity() {
  local entity_type="$1"
  local endpoint="$2"
  local data="$3"

  echo "Creating ${entity_type}..."
  set -x
  RESPONSE=$(curl -w "\nHTTP_CODE:%{http_code}" -sS --request POST \
    --header "PRIVATE-TOKEN: ${GITLAB_AUTO_RELEASE_CI}" \
    --header "Content-Type: application/json" \
    --data "${data}" \
    "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/${endpoint}" 2>&1 || true)
  set +x

  echo "Full ${entity_type} response:"
  echo "${RESPONSE}"

  HTTP_CODE=$(echo "${RESPONSE}" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
  echo "HTTP status code: ${HTTP_CODE}"

  if [ "$HTTP_CODE" != "201" ] && [ "$HTTP_CODE" != "200" ]; then
    echo "Failed to create ${entity_type}. HTTP code: ${HTTP_CODE}"
    exit 1
  fi
}

echo "Generating changelog..."
python sber_scripts/releases/generate_changelog.py

export RELEASE_VERSION=$(grep RELEASE_VERSION release.env | cut -d'=' -f2)
echo "Creating release version ${RELEASE_VERSION}"

RELEASE_NOTES=$(cat release_notes.md | python -c "import sys, json; print(json.dumps(sys.stdin.read()))")

echo "Debug - CI_API_V4_URL is ${CI_API_V4_URL}"
echo "Debug - CI_PROJECT_ID is ${CI_PROJECT_ID}"
echo "Debug - CI_COMMIT_SHA is ${CI_COMMIT_SHA}"
echo "Debug - Token present:" && [ -n "$GITLAB_AUTO_RELEASE_CI" ] && echo "Yes" || echo "No"

create_gitlab_entity "git tag v${RELEASE_VERSION}" "repository/tags" "{\"tag_name\":\"v${RELEASE_VERSION}\",\"ref\":\"${CI_COMMIT_SHA}\",\"message\":\"Release ${RELEASE_VERSION}\"}"

create_gitlab_entity "GitLab release" "releases" "{\"tag_name\":\"v${RELEASE_VERSION}\",\"name\":\"Release ${RELEASE_VERSION}\",\"description\":${RELEASE_NOTES}}"

echo "Release v${RELEASE_VERSION} created successfully!"
echo "Changelog and release notes are available in artifacts"
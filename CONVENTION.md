# MNIST Lab 협업 규칙

## 팀원 이니셜

| 이름 | 이니셜 |
|------|--------|
| 김영민 | ym |
| 백찬 | bc |
| 김민희 | mh |

## 브랜치

```
main                 ← 최종본
 └─ ym/dev           ← 개별 브랜치 (사람당 1개)
     └─ ym/day1-affine   ← 작업 브랜치 (작업마다 1개)
```

| 종류 | 이름 | 예 |
|------|------|----|
| 개별 브랜치 | `이니셜/dev` | `ym/dev`, `bc/dev`, `mh/dev` |
| 작업 브랜치 | `이니셜/day번호-작업내용` | `ym/day1-affine`, `mh/day2-softmax` |

- 개별 브랜치는 처음에 `main`에서 한 번 만든다
  ```bash
  git switch main && git pull
  git switch -c ym/dev && git push -u origin ym/dev
  ```
- 작업 브랜치는 **자기 개별 브랜치에서** 딴다
  ```bash
  git switch ym/dev && git pull
  git switch -c ym/day1-affine
  ```
- `main`, 남의 개별 브랜치에 직접 push 금지

## 통합 규칙

1. **작업 브랜치 → 개별 브랜치**: 작업이 끝나면 PR을 올려 자기 `이니셜/dev`에 merge
   - 셀프 merge 가능, `pytest`가 통과해야 함
   - merge 후 작업 브랜치 삭제
2. **개별 브랜치 → main**: 그날 작업을 마치면 `이니셜/dev` → `main` PR
   - 팀원 1명 이상 승인 후 merge
3. **main 최신화**: 작업 시작 전에 `main`을 개별 브랜치에 반영
   ```bash
   git switch ym/dev
   git pull origin main
   ```

## 커밋

```
[이니셜] 타입: 내용
```

| 타입 | 의미 |
|------|------|
| feat | 기능 구현 (TODO 채우기) |
| fix | 버그 수정 |
| test | 테스트 추가/수정 |
| docs | 문서, LEARNING_LOG, REPORT |
| refactor | 동작 변화 없는 코드 정리 |

- 예: `[ym] feat: Affine forward/backward 구현`

## PR

- 제목
  - 작업 → 개별: `[이니셜] Day번호 - 작업내용` (예: `[ym] Day1 - Affine 구현`)
  - 개별 → main: `[이니셜] Day번호 통합` (예: `[ym] Day1 통합`)
- PR을 올릴 때 **base 브랜치를 꼭 확인** (작업 브랜치 PR의 base는 `이니셜/dev`)
- 본문:
  ```
  ## 한 일
  -
  ## 확인
  - [ ] pytest tests/ 통과
  - [ ] python tools/selfcheck.py --day N 통과
  ```

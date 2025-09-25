파일을 깃헙과 연결된 파일에 넣은후 
# 1. 변경사항 확인
git status

# 2. 새 파일 포함한 모든 변경 스테이징
git add .

# 3. 커밋 메시지 작성
git commit -m "요구사항 및 유즈케이스 md 파일 추가"

# 4. 원격 변경 반영 (중요)
git pull --rebase origin main

# 5. 깃허브에 업로드
git push origin main

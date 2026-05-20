# GPU 서버 배포 가이드

이제 GPU 서버 배포 파일은 역할별 폴더로 분리되어 있습니다.

- `gpu-server/exaone`
  - EXAONE 텍스트 모델 서버
  - `exaone-vllm + redis`
- `gpu-server/vlm`
  - VLM 서버
  - `vlm-server + redis`

즉, 각 EC2에서 자기 역할 폴더만 사용하면 됩니다.

## 폴더 구조

```text
gpu-server/
  exaone/
    docker-compose.yaml
    .env.example
  vlm/
    docker-compose.yaml
    .env.example
  scripts/
    validate-host.sh
    validate-env.sh
    deploy.sh
    status.sh
    stop.sh
```

## 가장 중요한 사용법

### EXAONE 서버

EXAONE EC2에 접속해서 아래만 하면 됩니다.

```bash
cd /home/ubuntu/S14P31A401
cp gpu-server/exaone/.env.example gpu-server/exaone/.env
vi gpu-server/exaone/.env
docker compose -f gpu-server/exaone/docker-compose.yaml --env-file gpu-server/exaone/.env up -d
```

### VLM 서버

VLM EC2에 접속해서 아래만 하면 됩니다.

```bash
cd /home/ubuntu/S14P31A401
cp gpu-server/vlm/.env.example gpu-server/vlm/.env
vi gpu-server/vlm/.env
docker compose -f gpu-server/vlm/docker-compose.yaml --env-file gpu-server/vlm/.env up -d
```

즉, 이제는 예전처럼 공용 `gpu-server/docker-compose.yaml`에서 서비스명을 골라 띄우는 방식이 아닙니다.  
서버별 compose 파일이 이미 분리되어 있으므로 그냥 해당 폴더 compose를 `up -d` 하면 됩니다.

## EXAONE 서버 상세 절차

### 1. env 파일 생성

```bash
cp gpu-server/exaone/.env.example gpu-server/exaone/.env
```

필수 입력값:

- `HF_TOKEN`
- `REDIS_PASSWORD`

기본값:

```dotenv
EXAONE_VLLM_MODEL=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ
EXAONE_VLLM_PORT_PRIMARY=8000
EXAONE_VLLM_PORT_COMPAT=8002
EXAONE_VLLM_CTX=4096
EXAONE_VLLM_GPU_MEM=0.40
REDIS_PORT=6379
```

### 2. 직접 배포

```bash
docker compose \
  -f gpu-server/exaone/docker-compose.yaml \
  --env-file gpu-server/exaone/.env \
  up -d
```

### 3. 상태 확인

```bash
docker compose \
  -f gpu-server/exaone/docker-compose.yaml \
  --env-file gpu-server/exaone/.env \
  ps
```

```bash
curl http://127.0.0.1:8000/health
```

### 4. 중지

```bash
docker compose \
  -f gpu-server/exaone/docker-compose.yaml \
  --env-file gpu-server/exaone/.env \
  down
```

## VLM 서버 상세 절차

### 1. env 파일 생성

```bash
cp gpu-server/vlm/.env.example gpu-server/vlm/.env
```

필수 입력값:

- `HF_TOKEN`
- `REDIS_PASSWORD`

기본값:

```dotenv
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
VLM_PORT=8001
VLM_CTX=8192
VLM_GPU_MEM=0.45
REDIS_PORT=6379
```

### 2. 직접 배포

```bash
docker compose \
  -f gpu-server/vlm/docker-compose.yaml \
  --env-file gpu-server/vlm/.env \
  up -d
```

### 3. 상태 확인

```bash
docker compose \
  -f gpu-server/vlm/docker-compose.yaml \
  --env-file gpu-server/vlm/.env \
  ps
```

```bash
curl http://127.0.0.1:8001/health
```

### 4. 중지

```bash
docker compose \
  -f gpu-server/vlm/docker-compose.yaml \
  --env-file gpu-server/vlm/.env \
  down
```

## 스크립트를 쓰는 경우

스크립트는 필수가 아닙니다.  
그냥 `docker compose up -d`를 직접 써도 됩니다.

다만 아래를 자동화하고 싶으면 스크립트를 사용할 수 있습니다.

- env 검증
- host 점검
- health check
- Redis ping 확인

### EXAONE

```bash
bash gpu-server/scripts/validate-host.sh exaone
bash gpu-server/scripts/validate-env.sh exaone gpu-server/exaone/.env
bash gpu-server/scripts/deploy.sh exaone gpu-server/exaone/.env
bash gpu-server/scripts/status.sh exaone gpu-server/exaone/.env
```

### VLM

```bash
bash gpu-server/scripts/validate-host.sh vlm
bash gpu-server/scripts/validate-env.sh vlm gpu-server/vlm/.env
bash gpu-server/scripts/deploy.sh vlm gpu-server/vlm/.env
bash gpu-server/scripts/status.sh vlm gpu-server/vlm/.env
```

## 로그 확인

### EXAONE

```bash
docker compose \
  -f gpu-server/exaone/docker-compose.yaml \
  --env-file gpu-server/exaone/.env \
  logs --tail=200
```

### VLM

```bash
docker compose \
  -f gpu-server/vlm/docker-compose.yaml \
  --env-file gpu-server/vlm/.env \
  logs --tail=200
```

## 메인 애플리케이션 연결

메인 서버 `.env`는 아래처럼 넣습니다.

```dotenv
CHAT_MODEL_BASE_URL=http://<exaone-host>:8000
KEYWORD_MODEL_BASE_URL=http://<exaone-host>:8000
CAPTION_MODEL_BASE_URL=http://<exaone-host>:8000
FINAL_EDIT_MODEL_BASE_URL=http://<vlm-host>:8001
```

의미:

- 텍스트 계열 호출은 `exaone` 서버로 보냄
- 최종 편집 VLM 호출은 `vlm` 서버로 보냄

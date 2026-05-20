-- 1. 확장 모듈 및 사용자 정의 타입 생성 (사전 필수 작업)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- 도메인 요구사항에 맞게 ENUM 값 수정 필요
CREATE TYPE category_type AS ENUM ('주점', '제과점', '카페', '식당');
CREATE TYPE owner_persona_type AS ENUM ('aesthetic', 'friendly', 'professional', 'trendy', 'other');

-- ==============================================================================
-- 최상위 독립 테이블 (참조를 당하는 테이블)
-- ==============================================================================

-- =============================================================
--  USERS
--  Instagram 인증 사용자와 서비스 계정의 기본 정보를 저장한다.
-- =============================================================
CREATE TABLE "users" (
                         "id"                   UUID           NOT NULL,
                         "instagram_user_id"    VARCHAR(100)   NOT NULL,
                         "instagram_username"   VARCHAR(100)   NULL,
                         "access_token"         TEXT           NULL,
                         "token_expires_at"     TIMESTAMPTZ    NULL,
                         "created_at"           TIMESTAMPTZ    NULL,
                         "updated_at"           TIMESTAMPTZ    NULL,
                         "profile_image_url"    TEXT           NULL,
                         CONSTRAINT "PK_USERS" PRIMARY KEY ("id"), -- PK 추가
                         CONSTRAINT "UK_USERS_INSTAGRAM_USER_ID" UNIQUE ("instagram_user_id")
);

-- =============================================================
--  canonical_keywords
--  AI 추천/검색에서 사용할 표준 키워드와 임베딩을 저장한다.
-- =============================================================

CREATE TABLE "canonical_keywords" (
                                      "id"             BIGINT         NOT NULL,
                                      "code"           VARCHAR(100)   NULL,
                                      "display_name"   VARCHAR(200)   NOT NULL,
                                      "embedding"      VECTOR(384)    NULL,
                                      "created_at"     TIMESTAMPTZ    NOT NULL,
                                      CONSTRAINT "PK_CANONICAL_KEYWORDS" PRIMARY KEY ("id") -- PK 추가
);
COMMENT ON COLUMN "canonical_keywords"."display_name" IS '사용자/운영자가 읽는 표시 이름';
COMMENT ON COLUMN "canonical_keywords"."embedding" IS '표준 키워드 임베딩 벡터 (pgvector 코사인 유사도 검색용)';

-- =============================================================
--  REFERENCE_IMAGES
--  AI 이미지 추천에 사용할 레퍼런스 이미지 메타데이터와 임베딩을 저장한다.
-- =============================================================

CREATE TABLE "reference_images" (
                                    "id"                        BIGINT         NOT NULL,
                                    "sort_order"                SMALLINT       NULL,
                                    "s3_key"                    VARCHAR(200)   NOT NULL,
    -- 피사체
                                    "focus_subject"             VARCHAR(100)   NULL,
                                    "focus_subject_embedding"   VECTOR(384)    NULL,
    -- 촬영 구도
                                    "shot_type"                 VARCHAR(20)    NULL,
                                    "shot_type_code"            VARCHAR(50)    NULL,
                                    "shot_type_embedding"       VECTOR(384)   NULL,
    -- 카메라 앵글
                                    "camera_angle"              VARCHAR(20)    NULL,
                                    "camera_angle_code"         VARCHAR(50)    NULL,
                                    "camera_angle_embedding"    VECTOR(384)   NULL,
    -- 스타일/필터
                                    "style_filter"              VARCHAR(100)   NULL,
    -- 조명
                                    "lighting_params"           JSONB          NULL,
                                    CONSTRAINT "PK_REFERENCE_IMAGES" PRIMARY KEY ("id") -- PK 추가
);
COMMENT ON COLUMN "reference_images"."shot_type" IS '원본 텍스트 보존. 정규화 코드는 shot_type_code 사용.
shot_type_code 허용값: closeup | macro_detail | tabletop_medium | wide_full | flat_lay';

COMMENT ON COLUMN "reference_images"."camera_angle" IS '원본 텍스트 보존. 정규화 코드는 camera_angle_code 사용.
camera_angle_code 허용값: top_90 | high_60 | high_45 | high_30 | eye_level | dutch_angle | low_angle | worm_eye';

COMMENT ON COLUMN "reference_images"."style_filter" IS '예: 하이 컨트라스트/비비드, 페이디드/빈티지, 다크 시네마틱, 브라이트 앤 에어리, 앤 코지';

COMMENT ON COLUMN "reference_images"."lighting_params" IS '표준 키:
  type       → lowkey | softlight | natural | highlight
  direction  → front | side | back | top | spot
  color_temp → warm | cool | neutral
  mood       → cinematic | airy | moody | cozy
예시: {"type":"lowkey","direction":"spot","color_temp":"cool","mood":"cinematic"}';


-- =============================================================
--  REFERENCE_CAPTIONS
--  AI 문구 추천에 사용할 레퍼런스 캡션과 임베딩을 저장한다.
-- =============================================================

CREATE TABLE "reference_captions" (
                                      "id"                BIGINT         NOT NULL,
                                      "caption_content"   TEXT           NOT NULL,
                                      "embedding"         VECTOR(384)   NOT NULL,
                                      CONSTRAINT "PK_REFERENCE_CAPTIONS" PRIMARY KEY ("id") -- PK 추가
);

-- =============================================================
--  HOLIDAYS
--  마케팅 콘텐츠 생성 시 참고할 공휴일/기념일 정보를 저장한다.
-- =============================================================
CREATE TABLE "holidays" (
                            "id"             UUID           NOT NULL,
                            "holiday_date"   DATE           NOT NULL,
                            "name"           VARCHAR(200)   NOT NULL,
                            "created_at"     TIMESTAMPTZ    NULL,
                            CONSTRAINT "PK_HOLIDAYS" PRIMARY KEY ("id") -- PK 추가
);

-- ==============================================================================
-- 1차 종속 테이블 (users, reference 테이블 등 참조)
-- ==============================================================================

-- =============================================================
--  STORES
--  사용자 소유 매장의 기본 정보와 운영 정보를 저장한다.
-- =============================================================
CREATE TABLE "stores" (
                          "id"                UUID                 NOT NULL,
                          "user_id"           UUID                 NOT NULL,
                          "merchant_id"       VARCHAR(20)          NULL,
                          "store_name"        VARCHAR(200)         NOT NULL,
                          "category"          category_type        NULL,
                          "owner_persona"     owner_persona_type   NULL,
                          "address"           TEXT                 NULL,
                          "latitude"          DECIMAL(10, 7)       NULL,
                          "longitude"         DECIMAL(10, 7)       NULL,
                          "operating_hours"   JSONB                NULL,
                          "created_at"        TIMESTAMPTZ          NULL,
                          "updated_at"        TIMESTAMPTZ          NULL,
                          CONSTRAINT "PK_STORES" PRIMARY KEY ("id"),
                          CONSTRAINT "FK_users_TO_stores"
                              FOREIGN KEY ("user_id") REFERENCES "users"("id") -- FK 추가 (비식별 관계)
);
COMMENT ON COLUMN "stores"."operating_hours" IS '영업시간';

-- =============================================================
--  API_TOKEN_LOGS
--  Instagram API 토큰 갱신 이력을 저장한다.
-- =============================================================
CREATE TABLE "api_token_logs" (
                                  "id"               UUID          NOT NULL,
                                  "user_id"          UUID          NOT NULL,
                                  "old_expires_at"   TIMESTAMPTZ   NULL,
                                  "new_expires_at"   TIMESTAMPTZ   NULL,
                                  "created_at"       TIMESTAMPTZ   NULL,
                                  CONSTRAINT "PK_API_TOKEN_LOGS" PRIMARY KEY ("id"),
                                  CONSTRAINT "FK_users_TO_api_token_logs"
                                      FOREIGN KEY ("user_id") REFERENCES "users"("id") -- FK 추가
);

-- =============================================================
--  DEVICE_TOKENS
--  사용자별 FCM 디바이스 토큰과 활성 상태를 저장한다.
-- =============================================================
CREATE TABLE "device_tokens" (
                                 "id"           UUID          NOT NULL,
                                 "user_id"      UUID          NOT NULL,
                                 "token"        TEXT          NOT NULL,
                                 "platform"     VARCHAR(255)  NOT NULL,
                                 "is_active"    BOOLEAN       NOT NULL,
                                 "created_at"   TIMESTAMPTZ   NOT NULL,
                                 "updated_at"   TIMESTAMPTZ   NOT NULL,
                                 CONSTRAINT "PK_DEVICE_TOKENS" PRIMARY KEY ("id"),
                                 CONSTRAINT "UK_DEVICE_TOKENS_TOKEN" UNIQUE ("token"),
                                 CONSTRAINT "FK_users_TO_device_tokens"
                                     FOREIGN KEY ("user_id") REFERENCES "users"("id") -- FK 추가
);

-- =============================================================
--  REFERENCE
--  레퍼런스 캡션과 이미지의 게시물 단위 연결 정보를 저장한다.
-- =============================================================
CREATE TABLE "reference" (
                             "id"              BIGINT               NOT NULL,
                             "caption_id"      BIGINT               NOT NULL,
                             "image_id"        BIGINT               NOT NULL,
    -- 기존 maketing.owner_persona_type에서 스키마 분리를 제외하고 타입 사용으로 정규화
                             "owner_persona"   owner_persona_type   NULL,
                             "store_name"      VARCHAR(200)         NULL,
                             "created_at"      DATE                 NULL,
                             "instagram_id"    VARCHAR(200)         NULL,
                             CONSTRAINT "PK_REFERENCE" PRIMARY KEY ("id"),
                             CONSTRAINT "FK_reference_captions_TO_reference"
                                 FOREIGN KEY ("caption_id") REFERENCES "reference_captions"("id"), -- FK 추가
                             CONSTRAINT "FK_reference_images_TO_reference"
                                 FOREIGN KEY ("image_id") REFERENCES "reference_images"("id") -- FK 추가
);

-- ==============================================================================
-- 매핑 테이블 (식별 관계 적용 - 복합 기본키)
-- ==============================================================================

-- =============================================================
--  REFERENCE_IMAGE_KEYWORDS
--  레퍼런스 이미지와 표준 키워드의 다대다 매핑을 저장한다.
-- =============================================================
CREATE TABLE "reference_image_keywords" (
                                            "reference_image_id"     BIGINT   NOT NULL,
                                            "canonical_keyword_id"   BIGINT   NOT NULL,
    -- 식별 관계: 두 FK의 조합을 PK로 설정하여 N:M 매핑 테이블의 무결성 보장
                                            CONSTRAINT "PK_REFERENCE_IMAGE_KEYWORDS" PRIMARY KEY ("reference_image_id", "canonical_keyword_id"),
                                            CONSTRAINT "FK_reference_images_TO_reference_image_keywords"
                                                FOREIGN KEY ("reference_image_id") REFERENCES "reference_images"("id") ON DELETE CASCADE,
                                            CONSTRAINT "FK_canonical_keywords_TO_reference_image_keywords"
                                                FOREIGN KEY ("canonical_keyword_id") REFERENCES "canonical_keywords"("id") ON DELETE CASCADE
);

-- =============================================================
--  REFERENCE_CAPTION_KEYWORDS
--  레퍼런스 캡션과 표준 키워드의 다대다 매핑을 저장한다.
-- =============================================================
CREATE TABLE "reference_caption_keywords" (
                                              "reference_caption_id"   BIGINT   NOT NULL,
                                              "canonical_keyword_id"   BIGINT   NOT NULL,
    -- 식별 관계 적용
                                              CONSTRAINT "PK_REFERENCE_CAPTION_KEYWORDS" PRIMARY KEY ("reference_caption_id", "canonical_keyword_id"),
                                              CONSTRAINT "FK_reference_captions_TO_reference_caption_keywords"
                                                  FOREIGN KEY ("reference_caption_id") REFERENCES "reference_captions"("id") ON DELETE CASCADE,
                                              CONSTRAINT "FK_canonical_keywords_TO_reference_caption_keywords"
                                                  FOREIGN KEY ("canonical_keyword_id") REFERENCES "canonical_keywords"("id") ON DELETE CASCADE
);

-- ==============================================================================
-- 2차 종속 테이블 (stores 참조)
-- ==============================================================================

-- =============================================================
--  STORE_HOURS
--  매장별 요일 영업 시작/종료 시간을 저장한다.
-- =============================================================
CREATE TABLE "store_hours" (
                               "store_id"          UUID   NOT NULL,
                               "monday_open"       TIME   NULL,
                               "tuesday_open"      TIME   NULL,
                               "wednesday_open"    TIME   NULL,
                               "thursday_open"     TIME   NULL,
                               "friday_open"       TIME   NULL,
                               "saturday_open"     TIME   NULL,
                               "sunday_open"       TIME   NULL,
                               "monday_close"      TIME   NULL,
                               "tuesday_close"     TIME   NULL,
                               "wednesday_close"   TIME   NULL,
                               "thursday_close"    TIME   NULL,
                               "friday_close"      TIME   NULL,
                               "saturday_close"    TIME   NULL,
                               "sunday_close"      TIME   NULL,
    -- 식별 관계 (1:1): store_id를 PK이자 FK로 사용
                               CONSTRAINT "PK_STORE_HOURS" PRIMARY KEY ("store_id"),
                               CONSTRAINT "FK_stores_TO_store_hours"
                                   FOREIGN KEY ("store_id") REFERENCES "stores"("id") ON DELETE CASCADE
);

-- =============================================================
--  UPLOAD_HISTORY
--  매장별 콘텐츠 업로드 이력을 날짜/요일 기준으로 저장한다.
-- =============================================================
CREATE TABLE "upload_history" (
                                  "id"              UUID          NOT NULL,
                                  "store_id"        UUID          NOT NULL,
                                  "uploaded_at"     TIMESTAMPTZ   NOT NULL,
                                  "uploaded_date"   DATE          NULL,
                                  "day_of_week"     SMALLINT      NOT NULL,
                                  "created_at"      TIMESTAMPTZ   NULL,
                                  CONSTRAINT "PK_UPLOAD_HISTORY" PRIMARY KEY ("id"),
                                  CONSTRAINT "FK_stores_TO_upload_history"
                                      FOREIGN KEY ("store_id") REFERENCES "stores"("id") -- FK 추가
);

-- =============================================================
--  ACCOUNT_WEEKLY_METRICS
--  매장 계정의 주간 성과 지표와 목표 달성 정보를 저장한다.
-- =============================================================
CREATE TABLE "account_weekly_metrics" (
                                          "id"                   UUID            NOT NULL,
                                          "store_id"             UUID            NOT NULL,
                                          "week_start"           DATE            NULL,
                                          "total_reach"          INTEGER         NULL,
                                          "total_saves"          INTEGER         NULL,
                                          "total_shares"         INTEGER         NULL,
                                          "target_post_count"    INTEGER         NULL,
                                          "actual_post_count"    INTEGER         NULL,
                                          "achievement_rate"     DECIMAL(5, 2)   NULL,
                                          "visit_intent_score"   DECIMAL(5, 2)   NULL,
                                          "created_at"           TIMESTAMPTZ     NULL,
                                          "is_deleted"           BOOLEAN         NULL,
                                          CONSTRAINT "PK_ACCOUNT_WEEKLY_METRICS" PRIMARY KEY ("id"),
                                          CONSTRAINT "FK_stores_TO_account_weekly_metrics"
                                              FOREIGN KEY ("store_id") REFERENCES "stores"("id"), -- FK 추가
                                          CONSTRAINT "UK_account_weekly_metrics_store_week" UNIQUE ("store_id", "week_start")
);

-- =============================================================
--  UPLOAD_PATTERNS
--  매장별 업로드 패턴과 예측 업로드 시간을 저장한다.
-- =============================================================
CREATE TABLE "upload_patterns" (
                                   "id"               UUID          NOT NULL,
                                   "store_id"         UUID          NOT NULL,
                                   "day_of_week"      SMALLINT      NOT NULL,
                                   "predicted_time"   TIME          NULL,
                                   "sample_count"     INTEGER       NULL,
                                   "last_updated"     TIMESTAMPTZ   NULL,
                                   CONSTRAINT "PK_UPLOAD_PATTERNS" PRIMARY KEY ("id"),
                                   CONSTRAINT "FK_stores_TO_upload_patterns"
                                       FOREIGN KEY ("store_id") REFERENCES "stores"("id") -- FK 추가
);

-- =============================================================
--  MENUS
--  매장 메뉴 정보와 날씨/공휴일 태그, 메뉴 임베딩을 저장한다.
-- =============================================================
CREATE TABLE "menus" (
                         "id"             UUID           NOT NULL,
                         "store_id"       UUID           NOT NULL,
                         "name"           VARCHAR(200)   NOT NULL,
                         "price"          INTEGER        NULL,
                         "description"    TEXT           NULL,
                         "weather_tags"   JSONB DEFAULT '[]'::jsonb NULL, -- Default 빈 객체 형식 명시적 캐스팅
                         "holiday_tags"   JSONB DEFAULT '[]'::jsonb NULL,
                         "created_at"     TIMESTAMPTZ    NULL,
                         "updated_at"     TIMESTAMPTZ    NULL,
                         "embedding"      VECTOR(384)   NULL,
                         CONSTRAINT "PK_MENUS" PRIMARY KEY ("id"),
                         CONSTRAINT "FK_stores_TO_menus" FOREIGN KEY ("store_id") REFERENCES "stores"("id") -- FK 추가
);

-- =============================================================
--  NOTIFICATIONS
--  매장별 예약/즉시 알림의 상태와 발송 결과를 저장한다.
-- =============================================================
CREATE TABLE "notifications" (
                                 "id"               UUID          NOT NULL,
                                 "store_id"         UUID          NOT NULL,
                                 "notification"     TEXT          NOT NULL,
                                 "scheduled_at"     TIMESTAMPTZ   NOT NULL,
                                 "created_at"       TIMESTAMPTZ   NOT NULL,
                                 "type"             VARCHAR(50)   NOT NULL,
                                 "status"           VARCHAR(50)   NOT NULL,
                                 "sent_at"          TIMESTAMPTZ   NULL,
                                 "updated_at"       TIMESTAMPTZ   NOT NULL,
                                 "retry_count"      INTEGER       NOT NULL,
                                 "failure_reason"   TEXT          NULL,
                                 "web_url"          TEXT          NULL,
                                 "reference_id"     UUID          NULL,
                                 CONSTRAINT "PK_NOTIFICATIONS" PRIMARY KEY ("id"),
                                 CONSTRAINT "FK_stores_TO_notifications"
                                     FOREIGN KEY ("store_id") REFERENCES "stores"("id") -- FK 추가
);

-- =============================================================
--  CONTENTS
--  생성/발행된 콘텐츠 본문과 Instagram 게시 정보를 저장한다.
-- =============================================================
CREATE TABLE "contents" (
                            "id"                    BIGINT         GENERATED BY DEFAULT AS IDENTITY NOT NULL,
                            "store_id"              UUID           NOT NULL,
                            "session_id"            UUID           NULL,
                            "caption"               TEXT           NULL,
                            "instagram_media_id"    VARCHAR(100)   NULL,
                            "instagram_permalink"   TEXT           NULL,
                            "published_at"          TIMESTAMPTZ    NULL,
                            "is_deleted"            BOOLEAN        NULL,
                            "deleted_at"            TIMESTAMPTZ    NULL,
                            "created_at"            TIMESTAMPTZ    NULL,
                            CONSTRAINT "PK_CONTENTS" PRIMARY KEY ("id"),
                            CONSTRAINT "FK_stores_TO_contents"
                                FOREIGN KEY ("store_id") REFERENCES "stores"("id") -- FK 추가
);

-- ==============================================================================
-- 3차 종속 테이블 (contents 참조)
-- ==============================================================================

-- =============================================================
--  CONTENTS_IMAGES
--  콘텐츠에 연결된 이미지 S3 객체 키를 저장한다.
-- =============================================================
CREATE TABLE "contents_images" (
                                   "id"            UUID           NOT NULL,
                                   "contents_id"   BIGINT         NOT NULL,
                                   "s3_key"        VARCHAR(200)   NOT NULL,
                                   CONSTRAINT "PK_CONTENTS_IMAGES" PRIMARY KEY ("id"),
                                   CONSTRAINT "FK_contents_TO_contents_images"
                                       FOREIGN KEY ("contents_id") REFERENCES "contents"("id") ON DELETE CASCADE -- FK 추가
);

-- =============================================================
--  VIDEO_RECORDINGS
--  콘텐츠에 연결된 영상 녹화 파일의 S3 객체 키를 저장한다.
-- =============================================================
CREATE TABLE "video_recordings" (
                                    "id"            UUID           NOT NULL,
                                    "contents_id"   BIGINT         NOT NULL,
                                    "s3_key"        VARCHAR(200)   NOT NULL,
                                    CONSTRAINT "PK_VIDEO_RECORDINGS" PRIMARY KEY ("id"),
                                    CONSTRAINT "FK_contents_TO_video_recordings"
                                        FOREIGN KEY ("contents_id") REFERENCES "contents"("id") ON DELETE CASCADE -- FK 추가
);

-- =============================================================
--  INSTAGRAM_METRICS
--  Instagram 게시물별 도달/저장/공유/좋아요 지표를 저장한다.
-- =============================================================
CREATE TABLE "instagram_metrics" (
                                     "id"                   UUID           NOT NULL,
                                     "contents_id"          BIGINT         NULL,
                                     "store_id"             UUID           NOT NULL,
                                     "instagram_media_id"   VARCHAR(100)   NOT NULL,
                                     "week_start"           DATE           NULL,
                                     "week_end"             DATE           NULL,
                                     "reaches"              INTEGER        NULL,
                                     "saves"                INTEGER        NULL,
                                     "shares"               INTEGER        NULL,
                                     "likes"                INTEGER        NULL,
                                     "created_at"           TIMESTAMPTZ    NULL,
                                     CONSTRAINT "PK_INSTAGRAM_METRICS" PRIMARY KEY ("id"),
                                     CONSTRAINT "FK_contents_TO_instagram_metrics"
                                         FOREIGN KEY ("contents_id") REFERENCES "contents"("id"), -- FK 추가
                                     CONSTRAINT "FK_stores_TO_instagram_metrics"
                                         FOREIGN KEY ("store_id") REFERENCES "stores"("id"), -- FK 추가
                                     CONSTRAINT "UK_instagram_metrics_media_week" UNIQUE ("instagram_media_id", "week_start")
);

-- ==============================================================================
-- 인덱스 생성 (외래키 및 JSONB 검색 성능 최적화)
-- ==============================================================================

-- Foreign Key 인덱스 (JOIN 성능 확보)
CREATE INDEX idx_stores_user_id ON "stores" ("user_id");
CREATE INDEX idx_api_token_logs_user_id ON "api_token_logs" ("user_id");
CREATE INDEX idx_device_tokens_user_id ON "device_tokens" ("user_id");

CREATE INDEX idx_upload_history_store_id ON "upload_history" ("store_id");
CREATE INDEX idx_account_weekly_metrics_store_id ON "account_weekly_metrics" ("store_id");
CREATE INDEX idx_upload_patterns_store_id ON "upload_patterns" ("store_id");
CREATE INDEX idx_menus_store_id ON "menus" ("store_id");
CREATE INDEX idx_notifications_store_id ON "notifications" ("store_id");
CREATE INDEX idx_notifications_reference_id ON "notifications" ("reference_id");
CREATE INDEX idx_notifications_status_scheduled_at ON "notifications" ("status", "scheduled_at");
CREATE INDEX idx_contents_store_id ON "contents" ("store_id");

CREATE INDEX idx_contents_images_contents_id ON "contents_images" ("contents_id");
CREATE INDEX idx_video_recordings_contents_id ON "video_recordings" ("contents_id");
CREATE INDEX idx_instagram_metrics_contents_id ON "instagram_metrics" ("contents_id");
CREATE INDEX idx_instagram_metrics_store_id ON "instagram_metrics" ("store_id");

CREATE INDEX idx_reference_caption_id ON "reference" ("caption_id");
CREATE INDEX idx_reference_image_id ON "reference" ("image_id");

-- JSONB 인덱스 (JSON Key/Value 검색 성능 확보를 위한 GIN 인덱스)
CREATE INDEX idx_stores_operating_hours_gin ON "stores" USING GIN ("operating_hours");
CREATE INDEX idx_menus_weather_tags_gin ON "menus" USING GIN ("weather_tags");
CREATE INDEX idx_menus_holiday_tags_gin ON "menus" USING GIN ("holiday_tags");

ALTER TABLE notifications
    ALTER COLUMN type TYPE varchar(50)
        USING CASE type::text
                  WHEN '0' THEN 'REMIND'
                  WHEN '1' THEN 'WEATHER_MENU'
                  WHEN '2' THEN 'HOLIDAY_MENU'
                  WHEN '3' THEN 'HOLIDAY_OPERATION'
                  WHEN '4' THEN 'WEEKLY_STATS'
                  WHEN '5' THEN 'POSTING_SUCCESS'
                  WHEN '6' THEN 'POSTING_FAILED'
                  ELSE type::text
        END,
    ALTER COLUMN status TYPE varchar(50)
        USING CASE status::text
                  WHEN '0' THEN 'PENDING'
                  WHEN '1' THEN 'PROCESSING'
                  WHEN '2' THEN 'SENT'
                  WHEN '3' THEN 'FAILED'
                  ELSE status::text
        END;

ALTER TABLE account_weekly_metrics
    ADD COLUMN IF NOT EXISTS weekly_revenue NUMERIC(15,2) NULL,
    ADD COLUMN IF NOT EXISTS revenue_growth_rate NUMERIC(5,2) NULL,
    ADD COLUMN IF NOT EXISTS ai_strategy_suggestion TEXT NULL,
    ADD COLUMN IF NOT EXISTS is_strategy_executed BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN account_weekly_metrics.weekly_revenue IS '주간 매출';
COMMENT ON COLUMN account_weekly_metrics.revenue_growth_rate IS '매출 증가율';
COMMENT ON COLUMN account_weekly_metrics.ai_strategy_suggestion IS 'AI 전략 제안 내용';
COMMENT ON COLUMN account_weekly_metrics.is_strategy_executed IS '전략 실행 여부';

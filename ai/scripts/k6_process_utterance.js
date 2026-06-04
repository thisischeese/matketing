import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const STORE_ID = __ENV.STORE_ID || "00000000-0000-0000-0000-000000000001";
const OWNER_PERSONA = __ENV.OWNER_PERSONA || "aesthetic";
const UTTERANCE =
  __ENV.UTTERANCE || "오늘 비 오는 날에 어울리는 따뜻한 디저트 홍보 게시물 문구를 만들어줘";
const REQUEST_DATE = __ENV.REQUEST_DATE || "2026-05-29";
const THINK_TIME_SECONDS = Number(__ENV.THINK_TIME_SECONDS || "0");
const REQUEST_TIMEOUT = __ENV.REQUEST_TIMEOUT || "90s";
const SESSION_ID_PREFIX = __ENV.SESSION_ID_PREFIX || "loadtest";
const CONTEXT_ROUTING_VARIANT = __ENV.CONTEXT_ROUTING_VARIANT || "";

const processUtteranceLatency = new Trend("process_utterance_latency", true);
const timeoutRate = new Rate("timeout_rate");
const http5xxRate = new Rate("http_5xx_rate");
const processUtteranceSuccessRate = new Rate("process_utterance_success_rate");
const fallbackSuspectedCount = new Counter("fallback_suspected_count");

const parsedIterations = Number(__ENV.K6_ITERATIONS || "0");
const parsedVus = Number(__ENV.K6_VUS || "1");
const parsedDuration = __ENV.K6_DURATION || "60s";

export const options =
  parsedIterations > 0
    ? {
        vus: parsedVus,
        iterations: parsedIterations,
        thresholds: {
          http_req_failed: ["rate<0.05"],
          http_req_duration: ["p(95)<30000"],
          timeout_rate: ["rate<0.10"],
          http_5xx_rate: ["rate<0.05"],
        },
      }
    : {
        vus: parsedVus,
        duration: parsedDuration,
        thresholds: {
          http_req_failed: ["rate<0.05"],
          http_req_duration: ["p(95)<30000"],
          timeout_rate: ["rate<0.10"],
          http_5xx_rate: ["rate<0.05"],
        },
      };

function randomHex(length) {
  let output = "";
  for (let index = 0; index < length; index += 1) {
    output += Math.floor(Math.random() * 16).toString(16);
  }
  return output;
}

function uuidV4() {
  return [
    randomHex(8),
    randomHex(4),
    `4${randomHex(3)}`,
    `${(8 + Math.floor(Math.random() * 4)).toString(16)}${randomHex(3)}`,
    randomHex(12),
  ].join("-");
}

function buildWeatherPayload() {
  return {
    temperature: Number(__ENV.WEATHER_TEMPERATURE || "18"),
    precipitation: Number(__ENV.WEATHER_PRECIPITATION || "0.7"),
    cloud_cover: __ENV.WEATHER_CLOUD_COVER || "overcast",
    humidity: Number(__ENV.WEATHER_HUMIDITY || "72"),
    wind_speed: Number(__ENV.WEATHER_WIND_SPEED || "2.8"),
    pm10: Number(__ENV.WEATHER_PM10 || "28"),
    pm25: Number(__ENV.WEATHER_PM25 || "16"),
    diurnal_range: Number(__ENV.WEATHER_DIURNAL_RANGE || "8"),
    discomfort_index: Number(__ENV.WEATHER_DISCOMFORT_INDEX || "67"),
    heavy_rain_warning: __ENV.WEATHER_HEAVY_RAIN_WARNING || null,
    typhoon_warning: __ENV.WEATHER_TYPHOON_WARNING || null,
  };
}

function isTimeoutResponse(response) {
  return Boolean(response && (response.error || response.status === 0));
}

function looksLikeFallback(body) {
  if (!body || typeof body.caption !== "string" || typeof body.guide_text !== "string") {
    return false;
  }

  const fallbackMarkers = [
    "다시 시도해 주세요",
    "자연스럽게 소개해보세요",
    "가게의 분위기",
  ];

  return fallbackMarkers.some(
    (marker) => body.caption.includes(marker) || body.guide_text.includes(marker),
  );
}

export default function () {
  const sessionId = uuidV4();
  const headers = {
    "Content-Type": "application/json",
    "X-Load-Test-Run-Id": SESSION_ID_PREFIX,
  };
  if (CONTEXT_ROUTING_VARIANT) {
    headers["X-Context-Routing-Variant"] = CONTEXT_ROUTING_VARIANT;
  }
  const response = http.post(
    `${BASE_URL}/ai/sessions/${sessionId}/process-utterance`,
    JSON.stringify({
      store_id: STORE_ID,
      utterance: UTTERANCE,
      owner_persona: OWNER_PERSONA,
      date: REQUEST_DATE,
      weather: buildWeatherPayload(),
    }),
    {
      headers,
      tags: {
        endpoint: "process-utterance",
        scenario: "process-utterance",
        run_id: SESSION_ID_PREFIX,
      },
      timeout: REQUEST_TIMEOUT,
    },
  );

  processUtteranceLatency.add(response.timings.duration);

  const timedOut = isTimeoutResponse(response);
  const is5xx = response.status >= 500;
  const succeeded = response.status === 200;

  timeoutRate.add(timedOut);
  http5xxRate.add(is5xx);
  processUtteranceSuccessRate.add(succeeded);

  let body = null;
  try {
    body = response.json();
  } catch (_) {
    body = null;
  }

  if (looksLikeFallback(body)) {
    fallbackSuspectedCount.add(1);
  }

  check(response, {
    "status is 200": (res) => res.status === 200,
    "response has status field": () => body && typeof body.status === "string",
    "response has caption": () => body && typeof body.caption === "string",
    "response has guide_text": () => body && typeof body.guide_text === "string",
  });

  if (THINK_TIME_SECONDS > 0) {
    sleep(THINK_TIME_SECONDS);
  }
}

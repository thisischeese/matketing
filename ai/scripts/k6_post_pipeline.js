import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const STORE_ID = __ENV.STORE_ID || "00000000-0000-0000-0000-000000000001";
const OWNER_PERSONA = __ENV.OWNER_PERSONA || "aesthetic";
const UTTERANCE =
  __ENV.UTTERANCE || "카페 분위기가 잘 보이도록 오늘의 디저트 게시물 문구를 만들어줘";
const REQUEST_DATE = __ENV.REQUEST_DATE || "2026-05-29";
const THINK_TIME_SECONDS = Number(__ENV.THINK_TIME_SECONDS || "1");
const REQUEST_TIMEOUT = __ENV.REQUEST_TIMEOUT || "120s";
const SESSION_ID_PREFIX = __ENV.SESSION_ID_PREFIX || "loadtest";
const CONTEXT_ROUTING_VARIANT = __ENV.CONTEXT_ROUTING_VARIANT || "";
const DRAFT_KEYS = (__ENV.DRAFT_KEYS || "/ai-drafts/test-session/draft-001.png,/ai-drafts/test-session/draft-002.png")
  .split(",")
  .map((value) => value.trim())
  .filter((value) => value.length > 0)
  .slice(0, 3);

const pipelineLatency = new Trend("pipeline_latency", true);
const processUtteranceLatency = new Trend("pipeline_process_utterance_latency", true);
const finalEditLatency = new Trend("pipeline_final_edit_latency", true);
const timeoutRate = new Rate("timeout_rate");
const http5xxRate = new Rate("http_5xx_rate");
const pipelineSuccessRate = new Rate("pipeline_success_rate");
const pipelineCompletedCount = new Counter("pipeline_completed_count");
const finalEditResultCount = new Counter("final_edit_result_count");

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
    precipitation: Number(__ENV.WEATHER_PRECIPITATION || "0.5"),
    cloud_cover: __ENV.WEATHER_CLOUD_COVER || "partly_cloudy",
    humidity: Number(__ENV.WEATHER_HUMIDITY || "65"),
    wind_speed: Number(__ENV.WEATHER_WIND_SPEED || "2.1"),
    pm10: Number(__ENV.WEATHER_PM10 || "24"),
    pm25: Number(__ENV.WEATHER_PM25 || "13"),
    diurnal_range: Number(__ENV.WEATHER_DIURNAL_RANGE || "9"),
    discomfort_index: Number(__ENV.WEATHER_DISCOMFORT_INDEX || "64"),
    heavy_rain_warning: __ENV.WEATHER_HEAVY_RAIN_WARNING || null,
    typhoon_warning: __ENV.WEATHER_TYPHOON_WARNING || null,
  };
}

function isTimeoutResponse(response) {
  return Boolean(response && (response.error || response.status === 0));
}

export default function () {
  const sessionId = uuidV4();
  const startedAt = Date.now();
  const processHeaders = {
    "Content-Type": "application/json",
    "X-Load-Test-Run-Id": SESSION_ID_PREFIX,
  };
  if (CONTEXT_ROUTING_VARIANT) {
    processHeaders["X-Context-Routing-Variant"] = CONTEXT_ROUTING_VARIANT;
  }

  const processResponse = http.post(
    `${BASE_URL}/ai/sessions/${sessionId}/process-utterance`,
    JSON.stringify({
      store_id: STORE_ID,
      utterance: UTTERANCE,
      owner_persona: OWNER_PERSONA,
      date: REQUEST_DATE,
      weather: buildWeatherPayload(),
    }),
    {
      headers: processHeaders,
      tags: {
        endpoint: "process-utterance",
        scenario: "post-pipeline",
        run_id: SESSION_ID_PREFIX,
      },
      timeout: REQUEST_TIMEOUT,
    },
  );
  processUtteranceLatency.add(processResponse.timings.duration);

  let processBody = null;
  try {
    processBody = processResponse.json();
  } catch (_) {
    processBody = null;
  }

  check(processResponse, {
    "process status is 200": (res) => res.status === 200,
    "process response has caption": () =>
      processBody && typeof processBody.caption === "string",
  });

  if (processResponse.status !== 200) {
    timeoutRate.add(isTimeoutResponse(processResponse));
    http5xxRate.add(processResponse.status >= 500);
    pipelineSuccessRate.add(false);
    return;
  }

  if (THINK_TIME_SECONDS > 0) {
    sleep(THINK_TIME_SECONDS);
  }

  const finalEditResponse = http.post(
    `${BASE_URL}/ai/sessions/${sessionId}/final-edit`,
    JSON.stringify({
      session_id: sessionId,
      drafts: DRAFT_KEYS,
    }),
    {
      headers: {
        "Content-Type": "application/json",
        "X-Load-Test-Run-Id": SESSION_ID_PREFIX,
      },
      tags: {
        endpoint: "final-edit",
        scenario: "post-pipeline",
        run_id: SESSION_ID_PREFIX,
      },
      timeout: REQUEST_TIMEOUT,
    },
  );
  finalEditLatency.add(finalEditResponse.timings.duration);

  let finalBody = null;
  try {
    finalBody = finalEditResponse.json();
  } catch (_) {
    finalBody = null;
  }

  const completed =
    finalEditResponse.status === 200 &&
    finalBody &&
    Array.isArray(finalBody.results);

  timeoutRate.add(
    isTimeoutResponse(processResponse) || isTimeoutResponse(finalEditResponse),
  );
  http5xxRate.add(
    processResponse.status >= 500 || finalEditResponse.status >= 500,
  );
  pipelineSuccessRate.add(completed);

  if (completed) {
    pipelineCompletedCount.add(1);
    finalEditResultCount.add(finalBody.results.length);
    pipelineLatency.add(Date.now() - startedAt);
  }

  check(finalEditResponse, {
    "final-edit status is 200": (res) => res.status === 200,
    "final-edit response has results": () =>
      finalBody && Array.isArray(finalBody.results),
  });
}

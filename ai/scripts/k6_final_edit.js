import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const DRAFT_KEYS = (__ENV.DRAFT_KEYS || "/ai-drafts/test-session/draft-001.png")
  .split(",")
  .map((value) => value.trim())
  .filter((value) => value.length > 0)
  .slice(0, 3);
const THINK_TIME_SECONDS = Number(__ENV.THINK_TIME_SECONDS || "0");
const REQUEST_TIMEOUT = __ENV.REQUEST_TIMEOUT || "120s";
const SESSION_ID_PREFIX = __ENV.SESSION_ID_PREFIX || "loadtest";

const finalEditLatency = new Trend("final_edit_latency", true);
const timeoutRate = new Rate("timeout_rate");
const http5xxRate = new Rate("http_5xx_rate");
const finalEditSuccessRate = new Rate("final_edit_success_rate");
const finalEdit503Rate = new Rate("final_edit_503_rate");
const finalEditResultCount = new Counter("final_edit_result_count");
const firstResponseLatency = new Trend("final_edit_first_byte_latency", true);

const parsedIterations = Number(__ENV.K6_ITERATIONS || "0");
const parsedVus = Number(__ENV.K6_VUS || "4");
const parsedDuration = __ENV.K6_DURATION || "30s";

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

function isTimeoutResponse(response) {
  return Boolean(response && (response.error || response.status === 0));
}

export default function () {
  const sessionId = uuidV4();
  const response = http.post(
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
        scenario: "final-edit",
        run_id: SESSION_ID_PREFIX,
      },
      timeout: REQUEST_TIMEOUT,
    },
  );

  finalEditLatency.add(response.timings.duration);
  firstResponseLatency.add(response.timings.waiting);
  timeoutRate.add(isTimeoutResponse(response));
  http5xxRate.add(response.status >= 500);
  finalEditSuccessRate.add(response.status === 200);
  finalEdit503Rate.add(response.status === 503);

  let body = null;
  try {
    body = response.json();
  } catch (_) {
    body = null;
  }

  if (body && Array.isArray(body.results)) {
    finalEditResultCount.add(body.results.length);
  }

  check(response, {
    "status is 200": (res) => res.status === 200,
    "response has status field": () => body && typeof body.status === "string",
    "response has results array": () => body && Array.isArray(body.results),
  });

  if (THINK_TIME_SECONDS > 0) {
    sleep(THINK_TIME_SECONDS);
  }
}

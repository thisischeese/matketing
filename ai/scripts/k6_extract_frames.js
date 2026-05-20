import http from "k6/http";
import { check, sleep } from "k6";

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const VIDEO_KEY =
  __ENV.VIDEO_KEY || "/inputs/test-session/test-video.webm";
const THINK_TIME_SECONDS = Number(__ENV.THINK_TIME_SECONDS || "0");

export const options = {
  vus: Number(__ENV.K6_VUS || "4"),
  duration: __ENV.K6_DURATION || "30s",
  thresholds: {
    http_req_failed: ["rate<0.05"],
    http_req_duration: ["p(95)<30000"],
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

export default function () {
  const sessionId = uuidV4();
  const response = http.post(
    `${BASE_URL}/ai/sessions/${sessionId}/extract-frames`,
    JSON.stringify({
      session_id: sessionId,
      video: VIDEO_KEY,
    }),
    {
      headers: {
        "Content-Type": "application/json",
      },
      tags: {
        endpoint: "extract-frames",
      },
    },
  );

  check(response, {
    "status is 200": (res) => res.status === 200,
    "response has status field": (res) => {
      try {
        const body = res.json();
        return body && typeof body.status === "string";
      } catch {
        return false;
      }
    },
  });

  if (THINK_TIME_SECONDS > 0) {
    sleep(THINK_TIME_SECONDS);
  }
}

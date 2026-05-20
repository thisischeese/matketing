import { useEffect, useState } from 'react';
import CardShell from './CardShell';
import AcornStampTrue from '@/assets/stamp/AcornStampTrue.png';

export default function WeeklyStatCard({
  percent,
  plannedCount,
  achievedCount,
  aiStrategySuggestion,
  isStrategyExecuted = false,
  isStrategyExecuting = false,
  onExecuteStrategy,
}) {
  const [animatedPercent, setAnimatedPercent] = useState(0);

  useEffect(() => {
    const targetPercent = Number(percent);

    if (Number.isNaN(targetPercent)) {
      return;
    }

    let animationFrameId;
    let startTime;

    const duration = 900;

    const animate = (currentTime) => {
      if (!startTime) {
        startTime = currentTime;
      }

      const elapsedTime = currentTime - startTime;
      const progress = Math.min(elapsedTime / duration, 1);
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      const currentPercent = targetPercent * easedProgress;

      setAnimatedPercent(currentPercent);

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(animate);
      }
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationFrameId);
  }, [percent]);

  const displayPercent = animatedPercent.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });

  const progressPercent = Math.min(animatedPercent, 100);

  const radius = 85;
  const strokeWidth = 18;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset =
    circumference - (progressPercent / 100) * circumference;

  const hasStrategy = Boolean(aiStrategySuggestion);

  let statusTitle;
  let statusMessage;

  if (achievedCount === 0) {
    statusTitle = '주간 포스팅 현황';
    statusMessage = (
      <>
        목표 <strong className="font-semibold text-primary-100">{plannedCount}개</strong> 중 현재 <strong className="font-semibold text-primary-100">{achievedCount}개</strong> 달성!
        <br />
        기분 좋게 첫 포스팅을 시작해 볼까요?
      </>
    );
  } else if (achievedCount >= plannedCount) {
    statusTitle = '목표 달성 완료';
    statusMessage = (
      <>
        목표 <strong className="font-semibold text-primary-100">{plannedCount}개</strong> 중 <strong className="font-semibold text-primary-100">{achievedCount}개</strong> 달성!
        <br />
        이번 주 목표를 완벽하게 채우셨네요!
      </>
    );
  } else {
    statusTitle = '이번 주 달성률';
    statusMessage = (
      <>
        목표 <strong className="font-semibold text-primary-100">{plannedCount}개</strong> 중 현재 <strong className="font-semibold text-primary-100">{achievedCount}개</strong> 달성!
        <br />
        목표까지 조금만 더 힘내보세요!
      </>
    );
  }

  return (
    <CardShell className="flex flex-col items-center justify-center px-5 py-8">
      <p className="text-[18px] font-extrabold text-primary-100">
        이번 주 리포트
      </p>

      <h2 className="mt-2 text-center text-[24px] font-semibold tracking-tight text-accent-100">
        AI 마케팅 전략
      </h2>

      <button
        type="button"
        onClick={onExecuteStrategy}
        disabled={!hasStrategy || isStrategyExecuted || isStrategyExecuting}
        className="mt-8 flex min-h-[110px] items-center justify-center"
      >
        {isStrategyExecuted ? (
          <img
            src={AcornStampTrue}
            alt="전략 실행 완료"
            className="h-[108px] w-[108px] object-contain"
          />
        ) : (
          <div className="h-[108px] w-[108px] rounded-full border-2 border-dashed border-gray-200" />
        )}
      </button>

      <p className="mt-3 text-center text-[12px] leading-relaxed text-gray-300">
        {isStrategyExecuted
          ? '이번 주 전략을 완료하셨어요'
          : '실행하면 이번 주 카드에 도장이 찍혀요'}
        <br />
        다음 주 월요일에 새로운 전략이 도착해요
      </p>

      <div className="mt-5 w-full rounded-[18px] bg-surface-100 px-5 py-5 text-center">
        <p className="whitespace-pre-line text-[17px] font-normal leading-relaxed text-gray-800">
          {hasStrategy
            ? aiStrategySuggestion
            : '아직 추천 전략이 없어요.\n지난주 데이터가 쌓이면 전략을 받아볼 수 있어요.'}
        </p>
      </div>

      <h2 className="mt-20 text-center text-[24px] font-semibold tracking-tight text-accent-100">
        {statusTitle}
      </h2>

      <div className="relative mt-6 flex h-[190px] w-[190px] items-center justify-center">
        <svg
          width="190"
          height="190"
          viewBox="0 0 190 190"
          className="-rotate-90 transform"
          aria-hidden="true"
        >
          <circle
            cx="95"
            cy="95"
            r={normalizedRadius}
            fill="transparent"
            stroke="var(--color-gray-100)"
            strokeWidth={strokeWidth}
          />
          <circle
            cx="95"
            cy="95"
            r={normalizedRadius}
            fill="transparent"
            stroke="var(--color-primary-100)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <strong className="text-[50px] font-semibold tracking-tighter text-accent-100">
            {displayPercent}
            <span className="ml-1 text-[25px] font-medium text-gray-500">
              %
            </span>
          </strong>
        </div>
      </div>

      <p className="text-center mt-2 text-[17px] font-normal leading-relaxed text-gray-500">
        {statusMessage}
      </p>
    </CardShell>
  );
}

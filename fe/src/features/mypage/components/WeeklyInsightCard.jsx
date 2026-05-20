import { useEffect, useState } from 'react';
import Modal from '@/components/common/Modal';
import CardShell from './CardShell';

const DAY_LABELS = ['일', '월', '화', '수', '목', '금', '토'];

const DESCRIPTION_CONTENT = {
  reach: {
    title: '가게 노출수',
    content: `지난주 인스타그램 계정의 노출 수예요.\n수치가 높을수록 더 많은 사람에게\n가게가 보여진 거예요.`,
  },
  visitIntent: {
    title: '방문 관심도',
    content: `게시물을 본 사람들이\n가게에 관심을 보인 정도예요.\n저장, 공유, 노출 수를 반영해\n계산해요.`,
  },
};

const formatWeekRange = (weekStart, weekEnd) => {
  if (!weekStart || !weekEnd) {
    return '';
  }

  const startDate = new Date(weekStart);
  const endDate = new Date(weekEnd);

  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return '';
  }

  const startMonth = startDate.getMonth() + 1;
  const startDay = startDate.getDate();
  const startDayLabel = DAY_LABELS[startDate.getDay()];

  const endMonth = endDate.getMonth() + 1;
  const endDay = endDate.getDate();
  const endDayLabel = DAY_LABELS[endDate.getDay()];

  return `${startMonth}월 ${startDay}일 (${startDayLabel}) ~ ${endMonth}월 ${endDay}일 (${endDayLabel})`;
};

function useAnimatedNumber(value) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const targetValue = Number(value);

    if (Number.isNaN(targetValue)) {
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
      const currentValue = targetValue * easedProgress;

      setAnimatedValue(currentValue);

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(animate);
      }
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationFrameId);
  }, [value]);

  return animatedValue;
}

function InsightMetricItem({
  title,
  value,
  unit,
  descriptionKey,
  valuePrefix,
  valueColorClassName = 'text-primary-100',
  decimalPlaces = 0,
  onDescriptionOpen,
}) {
  const animatedValue = useAnimatedNumber(value);

  const displayValue = animatedValue.toLocaleString(undefined, {
    minimumFractionDigits: decimalPlaces,
    maximumFractionDigits: decimalPlaces,
  });

  return (
    <div className="flex flex-1 flex-col items-center gap-3 px-2">
      <div className="flex items-center justify-center gap-1.5">
        <h3 className="text-[16px] font-semibold tracking-tight text-accent-100">
          {title}
        </h3>

        <button
          type="button"
          onClick={() => onDescriptionOpen(descriptionKey)}
          className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-gray-200 text-gray-500 active:scale-95 active:bg-gray-300"
          aria-label={`${title} 설명 보기`}
        >
          <span className="material-icons" style={{ fontSize: '13px' }}>
            question_mark
          </span>
        </button>
      </div>

      <div className="mt-2 flex items-baseline justify-center gap-1">
        {valuePrefix && (
          <span className={`text-[16px] font-bold ${valueColorClassName}`}>
            {valuePrefix}
          </span>
        )}

        <strong
          className={`text-[28px] font-extrabold tracking-tighter ${valueColorClassName}`}
        >
          {displayValue}
        </strong>

        {unit && (
          <span className={`text-[16px] font-medium ${valueColorClassName}`}>
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

export default function WeeklyInsightCard({
  weekStart,
  weekEnd,
  reachCount,
  visitIntentScore,
}) {
  const [selectedDescription, setSelectedDescription] = useState(null);

  const weekRangeText = formatWeekRange(weekStart, weekEnd);
  const description = selectedDescription
    ? DESCRIPTION_CONTENT[selectedDescription]
    : null;

  const visitIntentDiffRate = Number(visitIntentScore) || 0;
  const visitIntentPrefix =
    visitIntentDiffRate > 0 ? '▲' : visitIntentDiffRate < 0 ? '▼' : '';
  const visitIntentValue = Math.abs(visitIntentDiffRate);
  const visitIntentColorClassName =
    visitIntentDiffRate < 0 ? 'text-gray-500' : 'text-primary-100';

  return (
    <>
      <CardShell className="flex flex-col items-center px-5 py-8">
        <p className="text-center text-[18px] font-extrabold text-primary-100">
          지난 주 리포트
        </p>

        {weekRangeText && (
          <div className="mt-4 mb-2 text-center">
            <span className="inline-block rounded-full bg-surface-100 px-3.5 py-1.5 text-[14px] font-semibold tracking-tight text-primary-100">
              {weekRangeText}
            </span>
          </div>
        )}

        <div className="mt-4 flex w-full items-stretch justify-center">
          <InsightMetricItem
            title="가게 노출수"
            value={reachCount}
            unit="회"
            descriptionKey="reach"
            valueColorClassName="text-accent-100"
            onDescriptionOpen={setSelectedDescription}
          />

          <div className="mx-2 w-px self-stretch bg-gray-100" />

          <InsightMetricItem
            title="방문 관심도"
            value={visitIntentValue}
            valuePrefix={visitIntentPrefix}
            unit="%"
            descriptionKey="visitIntent"
            decimalPlaces={1}
            valueColorClassName={visitIntentColorClassName}
            onDescriptionOpen={setSelectedDescription}
          />
        </div>
      </CardShell>

      <Modal
        isOpen={Boolean(selectedDescription)}
        onClose={() => setSelectedDescription(null)}
        title={description?.title}
        description={description?.content}
        titleColor="primary"
      />
    </>
  );
}

import WeeklyStatCard from '../components/WeeklyStatCard';
import WeeklyInsightCard from '../components/WeeklyInsightCard';

export default function MyPageStatsSection({
  reachWeekStart,
  reachWeekEnd,
  weeklyPostAchievementRate,
  targetPostCount,
  achievedPostCount,
  weeklyReachCount,
  weeklyVisitIntentScore,
  aiStrategySuggestion,
  isStrategyExecuted,
  isStrategyExecuting,
  onExecuteStrategy,
}) {
  return (
    <section className="flex flex-col gap-4">
      <WeeklyStatCard
        percent={weeklyPostAchievementRate}
        plannedCount={targetPostCount}
        achievedCount={achievedPostCount}
        aiStrategySuggestion={aiStrategySuggestion}
        isStrategyExecuted={isStrategyExecuted}
        isStrategyExecuting={isStrategyExecuting}
        onExecuteStrategy={onExecuteStrategy}
      />

      <WeeklyInsightCard
        weekStart={reachWeekStart}
        weekEnd={reachWeekEnd}
        reachCount={weeklyReachCount}
        visitIntentScore={weeklyVisitIntentScore}
      />
    </section>
  );
}

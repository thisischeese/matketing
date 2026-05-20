import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import BottomTab from '@/components/common/BottomTab';
import Modal from '@/components/common/Modal';
import MyPageHeader from '@/features/mypage/components/MyPageHeader';
import MyPageTabSwitcher from '@/features/mypage/components/MyPageTabSwitcher';
import MyPageStatsSection from '@/features/mypage/sections/MyPageStatsSection';
import AccountInfoSection from '@/features/mypage/sections/AccountInfoSection';
import OperatingHoursSection from '@/features/mypage/sections/OperatingHoursSection';
import { mypageApi } from '@/features/mypage/api';
import { authApi } from '@/features/auth/api';

const clearAuthStorage = () => {
  localStorage.removeItem('accessToken');
  sessionStorage.removeItem('instagramAuthPurpose');
};

const DEFAULT_ANALYTICS_DATA = {
  reachWeekStart: '',
  reachWeekEnd: '',
  weeklyPostAchievementRate: 0,
  targetPostCount: 0,
  achievedPostCount: 0,
  weeklyReachCount: 0,
  weeklyVisitIntentScore: 0,
  aiStrategySuggestion: null,
  isStrategyExecuted: false,
};

export default function MyPage() {
  const navigate = useNavigate();

  const [activeTab, setActiveTab] = useState('stats');
  const [myInfo, setMyInfo] = useState(null);
  const [analyticsData, setAnalyticsData] = useState(DEFAULT_ANALYTICS_DATA);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [isStrategyExecuting, setIsStrategyExecuting] = useState(false);
  const [isEditWarningModalOpen, setIsEditWarningModalOpen] = useState(false);

  const fetchMyInfo = async () => {
    try {
      const response = await mypageApi.getMyInfo();

      setMyInfo(response.data);
    } catch (error) {
      console.error('마이페이지 정보 조회 실패:', error);
    }
  };

  useEffect(() => {
    let isMounted = true;

    const loadMyPageData = async () => {
      try {
        const myInfoResponse = await mypageApi.getMyInfo();

        if (!isMounted) return;

        setMyInfo(myInfoResponse.data);
      } catch (error) {
        console.error('마이페이지 정보 조회 실패:', error);
        return;
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }

      try {
        const [lastWeekResponse, thisWeekResponse] = await Promise.all([
          mypageApi.getLastWeekAnalytics(),
          mypageApi.getThisWeekAnalytics(),
        ]);

        if (!isMounted) return;

        setAnalyticsData({
          reachWeekStart: lastWeekResponse.data.week_start ?? '',
          reachWeekEnd: lastWeekResponse.data.week_end ?? '',
          weeklyPostAchievementRate:
            thisWeekResponse.data.achievement_rate ?? 0,
          targetPostCount:
            thisWeekResponse.data.target_post_count ?? 0,
          achievedPostCount:
            thisWeekResponse.data.actual_post_count ?? 0,
          weeklyReachCount:
            lastWeekResponse.data.total_reach ?? 0,
          weeklyVisitIntentScore:
            lastWeekResponse.data.visit_intent_score ?? 0,
          aiStrategySuggestion:
            lastWeekResponse.data.ai_strategy_suggestion ?? null,
          isStrategyExecuted:
            thisWeekResponse.data.is_strategy_executed ?? false,
        });
      } catch (error) {
        console.error('마이페이지 통계 조회 실패:', error);
      }
    };

    loadMyPageData();

    return () => {
      isMounted = false;
    };
  }, []);

  const user = myInfo?.user;
  const store = myInfo?.store;

  const storeName = store?.storeName ?? '매장명 없음';

  const instagramUsername = user?.instagramUsername
    ? `@${user.instagramUsername}`
    : '@instagram';

  const handleTabChange = (nextTab) => {
    if (isEditing) {
      setIsEditWarningModalOpen(true);
      return;
    }

    setActiveTab(nextTab);
  };

  const handleCloseEditWarningModal = () => {
    setIsEditWarningModalOpen(false);
  };

  const handleExecuteStrategy = async () => {
    if (analyticsData.isStrategyExecuted || isStrategyExecuting) return;

    setIsStrategyExecuting(true);

    try {
      await mypageApi.executeWeeklyStrategy();

      setAnalyticsData((prev) => ({
        ...prev,
        isStrategyExecuted: true,
      }));
    } catch (error) {
      console.error('전략 실행 완료 처리 실패:', error);
      alert('전략 실행 완료 처리에 실패했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setIsStrategyExecuting(false);
    }
  };

  const handleLogout = async () => {
    try {
      await authApi.logout();
      clearAuthStorage();
      navigate('/', { replace: true });
    } catch (error) {
      console.error('로그아웃 API 호출 실패:', error);
      alert('로그아웃에 실패했습니다. 잠시 후 다시 시도해주세요.');
    }
  };

  const handleDeleteAccount = async () => {
    try {
      await authApi.deleteAccount();
    } catch (error) {
      console.error('회원탈퇴 API 호출 실패:', error);
      throw error;
    }

    clearAuthStorage();
    navigate('/', { replace: true });
  };

  const handleInstagramUpdate = async () => {
    try {
      const response = await mypageApi.syncInstagramProfile();
      const syncedProfile = response.data.data;

      setMyInfo((prev) => ({
        ...prev,
        user: {
          ...prev.user,
          instagramUserId: syncedProfile.instagramUserId,
          instagramUsername: syncedProfile.instagramUsername,
          profileImageUrl: syncedProfile.profileImageUrl,
        },
      }));
    } catch (error) {
      console.error('인스타그램 프로필 동기화 실패:', error);
      throw error;
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-dvh items-center justify-center bg-[#f4f4f6] text-base text-gray-500">
        마이페이지 정보를 불러오는 중입니다.
      </div>
    );
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-[#f4f4f6]">
      <MyPageHeader
        storeName={storeName}
        instagramUsername={instagramUsername}
        profileImageUrl={user?.profileImageUrl}
        onInstagramUpdate={handleInstagramUpdate}
      />

      <MyPageTabSwitcher activeTab={activeTab} onChange={handleTabChange} />

      <main className="mx-auto flex min-h-0 w-full max-w-[430px] flex-1 flex-col gap-2 overflow-y-auto px-5 pt-4 pb-5">
        {activeTab === 'stats' && (
          <MyPageStatsSection
            reachWeekStart={analyticsData.reachWeekStart}
            reachWeekEnd={analyticsData.reachWeekEnd}
            weeklyPostAchievementRate={analyticsData.weeklyPostAchievementRate}
            targetPostCount={analyticsData.targetPostCount}
            achievedPostCount={analyticsData.achievedPostCount}
            weeklyReachCount={analyticsData.weeklyReachCount}
            weeklyVisitIntentScore={analyticsData.weeklyVisitIntentScore}
            aiStrategySuggestion={analyticsData.aiStrategySuggestion}
            isStrategyExecuted={analyticsData.isStrategyExecuted}
            isStrategyExecuting={isStrategyExecuting}
            onExecuteStrategy={handleExecuteStrategy}
          />
        )}

        {activeTab === 'info' && (
          <AccountInfoSection
            store={store}
            onLogout={handleLogout}
            onRefresh={fetchMyInfo}
            onDeleteAccount={handleDeleteAccount}
            onEditingChange={setIsEditing}
          />
        )}

        {activeTab === 'hours' && (
          <OperatingHoursSection
            operatingHours={store?.operatingHours}
            onRefresh={fetchMyInfo}
            onEditingChange={setIsEditing}
          />
        )}
      </main>

      <BottomTab fixed={false} />

      {isEditWarningModalOpen && (
        <Modal
          isOpen={isEditWarningModalOpen}
          onClose={handleCloseEditWarningModal}
          showClose={false}
          title="수정 중인 내용이 있어요"
          description={`저장하거나 취소한 뒤\n다른 화면으로 이동해주세요.`}
          titleColor="primary"
          confirmText="확인"
          onConfirm={handleCloseEditWarningModal}
        />
      )}
    </div>
  );
}

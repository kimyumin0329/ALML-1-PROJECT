import React, { useMemo } from 'react';

function safe(n) {
  const v = Number(n || 0);
  return Number.isFinite(v) ? v : 0;
}

export default function NotificationStatusChart({ stats }) {
  const data = useMemo(() => {
    const total = safe(stats?.total);
    const danger = safe(stats?.DANGER);
    const warning = safe(stats?.WARNING);
    const safeCnt = safe(stats?.SAFE);
    const unknown = safe(stats?.UNKNOWN);

    const denom = total > 0 ? total : Math.max(1, danger + warning + safeCnt + unknown);

    const toPct = (x) => Math.round((x / denom) * 100);

    return [
      { key: 'DANGER', label: '위험', value: danger, pct: toPct(danger) },
      { key: 'WARNING', label: '주의', value: warning, pct: toPct(warning) },
      { key: 'SAFE', label: '안전', value: safeCnt, pct: toPct(safeCnt) },
      { key: 'UNKNOWN', label: '기타', value: unknown, pct: toPct(unknown) },
    ];
  }, [stats]);

  return (
    <div className="notification-status-chart">
      <div style={{ marginBottom: 8 }}>
        <strong>알림 통계</strong> (총 {safe(stats?.total)}건)
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {data.map((d) => (
          <div key={d.key}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
              <span>{d.label}</span>
              <span>
                {d.value} ({d.pct}%)
              </span>
            </div>

            <div style={{ height: 10, borderRadius: 6, background: '#1f2937' }}>
              <div
                style={{
                  height: 10,
                  width: `${Math.max(0, Math.min(100, d.pct))}%`,
                  borderRadius: 6,
                  background: '#9ca3af',
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * 파일 이름 : SettingFeedbackPage.jsx
 * 기능 : 문의/피드백 제출 페이지. 버그 제보나 문의 내용을 전송할 수 있다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useState } from 'react';
import { sendFeedback } from '../../api/SettingApi';

/**
 * 함수 이름 : SettingFeedbackPage
 * 기능 : 피드백 제출 페이지 컴포넌트.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 피드백 제출 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function SettingFeedbackPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [content, setContent] = useState('');

  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!content.trim()) {
      alert('내용을 입력해주세요.');
      return;
    }

    setLoading(true);
    setNote('');

    try {
      const res = await sendFeedback({ name, email, content });
      setNote(`전송 완료${res?.ticketId ? ` (ticketId: ${res.ticketId})` : ''}`);
      setContent('');
    } catch (e2) {
      setNote('전송 실패(백엔드 미구현/오프라인 가능). 내용은 복사해서 전달해 주세요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1>문의 / 피드백</h1>
      <p style={{ fontSize: 14, color: '#9ca3af', marginBottom: 16 }}>
        버그 제보/문의 내용을 전송합니다.
      </p>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 520 }}>
        <label>
          이름(선택)
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>

        <label>
          이메일(선택)
          <input value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>

        <label>
          내용
          <textarea
            rows={6}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="문제 상황/재현 방법/스크린샷 설명 등을 적어주세요."
          />
        </label>

        <button className="btn" type="submit" disabled={loading}>
          {loading ? '전송 중...' : '제출'}
        </button>

        {note && <p style={{ fontSize: 13, color: '#6b7280' }}>{note}</p>}
      </form>
    </div>
  );
}

export default SettingFeedbackPage;

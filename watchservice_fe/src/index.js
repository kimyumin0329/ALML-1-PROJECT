/**
 * 파일 이름 : index.js
 * 기능 : 애플리케이션의 진입점. React 애플리케이션을 DOM에 렌더링한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import './App.css';
import './styles/layout.css';
import './styles/component.css';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

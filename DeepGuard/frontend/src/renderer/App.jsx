// frontend/src/renderer/App.jsx
// Root renderer component. Switches between the app's two main views.

import { useState } from 'react';

const TABS = [
  { id: 'upload', label: 'Upload File' },
  { id: 'history', label: 'History' },
];

function App() {
  const [activeTab, setActiveTab] = useState('upload');

  return (
    <div className="app">
      <nav role="tablist" aria-label="Detection mode">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            className={activeTab === tab.id ? 'tab tab--active' : 'tab'}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="app__content">
        <section
          id="panel-upload"
          role="tabpanel"
          aria-labelledby="tab-upload"
          hidden={activeTab !== 'upload'}
        >
          Upload File view (future issue)
        </section>

        <section
          id="panel-history"
          role="tabpanel"
          aria-labelledby="tab-history"
          hidden={activeTab !== 'history'}
        >
          History view (future issue)
        </section>
      </main>
    </div>
  );
}

export default App;
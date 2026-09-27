// frontend/src/renderer/components/UploadArea.jsx
// Drag-and-drop area for selecting an image or video to scan.
//
// Note: this component always hands the raw browser File object to
// useDetection's detect(). It never resolves a filesystem path itself —
// that conversion (via window.api.getPathForFile, since Electron 32+ removed
// File.path) is the hook's responsibility, not this component's.

import { useState, useRef, useEffect } from 'react';

import { useDetection } from '../hooks/useDetection';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'video/mp4'];

function isAccepted(file) {
  return ACCEPTED_TYPES.includes(file.type);
}

function UploadArea() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const inputRef = useRef(null);

  const { detect, isDetecting, result, error } = useDetection();

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(nextFile) {
    if (nextFile && isAccepted(nextFile)) {
      setFile(nextFile);
    }
  }

  function openFileDialog() {
    if (inputRef.current) {
      inputRef.current.click();
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragActive(false);

    const dropped = event.dataTransfer && event.dataTransfer.files
      ? event.dataTransfer.files[0]
      : null;
    if (dropped) {
      selectFile(dropped);
    }
  }

  return (
    <div className="upload-area">
      <div
        className={dragActive ? 'dropzone dropzone--active' : 'dropzone'}
        role="button"
        tabIndex={0}
        aria-label="Upload an image or video"
        onClick={openFileDialog}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            openFileDialog();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          hidden
          onChange={(event) => {
            const selected = event.target.files && event.target.files[0];
            if (selected) {
              selectFile(selected);
            }
          }}
        />

        {previewUrl ? (
          file.type.startsWith('video/') ? (
            <video src={previewUrl} controls className="upload-area__preview" />
          ) : (
            <img
              src={previewUrl}
              alt="Selected file preview"
              className="upload-area__preview"
            />
          )
        ) : (
          <p className="dropzone__hint">
            Drag and drop an image or video here, or click to browse.
          </p>
        )}
      </div>

      <button
        type="button"
        className="detect-button"
        disabled={!file || isDetecting}
        onClick={() => detect(file)}
      >
        {isDetecting ? 'Detecting…' : 'Detect'}
      </button>

      {error && (
        <p className="upload-area__error" role="alert">
          {error}
        </p>
      )}

      {result && !error && (
        <p className="upload-area__result">
          Result: {result.label} ({Math.round(result.confidence * 100)}%)
        </p>
      )}
    </div>
  );
}

export default UploadArea;
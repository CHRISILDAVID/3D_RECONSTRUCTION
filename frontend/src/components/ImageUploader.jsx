import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

const MIN_IMAGES = 2
const MAX_IMAGES = 300

export default function ImageUploader({ onUpload, segment, onSegmentChange }) {
    const [files, setFiles] = useState([])
    const [uploading, setUploading] = useState(false)

    const onDrop = useCallback((acceptedFiles) => {
        const newFiles = [...files, ...acceptedFiles].slice(0, MAX_IMAGES)
        setFiles(newFiles)
    }, [files])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'],
        },
        multiple: true,
    })

    const removeFile = (index) => {
        setFiles(prev => prev.filter((_, i) => i !== index))
    }

    const handleUpload = async () => {
        if (files.length < MIN_IMAGES) return
        setUploading(true)
        try {
            await onUpload(files)
        } catch {
            setUploading(false)
        }
    }

    return (
        <div className="upload-section">
            <div className="upload-hero">
                <h2 className="section-title">
                    <span className="gradient-text">Reconstruct</span> Your Scene in 3D
                </h2>
                <p className="section-desc">
                    Upload multiple photos of the same scene from different angles.
                    Our AI will reconstruct a 3D point cloud of the environment.
                </p>
            </div>

            {/* Dropzone */}
            <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${files.length > 0 ? 'dropzone-has-files' : ''}`}
            >
                <input {...getInputProps()} />
                <div className="dropzone-content">
                    <div className="dropzone-icon">
                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                            <path d="M24 4L36 16H28V28H20V16H12L24 4Z" fill="url(#upGrad)" opacity="0.8" />
                            <path d="M8 34V40C8 41.1 8.9 42 10 42H38C39.1 42 40 41.1 40 40V34" stroke="url(#upGrad)" strokeWidth="2.5" strokeLinecap="round" />
                            <defs>
                                <linearGradient id="upGrad" x1="8" y1="4" x2="40" y2="42">
                                    <stop stopColor="#818cf8" />
                                    <stop offset="1" stopColor="#c084fc" />
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>
                    {isDragActive ? (
                        <p className="dropzone-text">Drop images here...</p>
                    ) : (
                        <>
                            <p className="dropzone-text">Drag & drop images here</p>
                            <p className="dropzone-hint">or click to browse · JPG, PNG, WebP supported</p>
                        </>
                    )}
                </div>
            </div>

            {/* Image previews */}
            {files.length > 0 && (
                <div className="preview-section">
                    <div className="preview-header">
                        <span className="preview-count">
                            <span className="count-number">{files.length}</span>
                            <span className="count-label"> image{files.length !== 1 ? 's' : ''} selected</span>
                        </span>
                        <button className="btn btn-ghost" onClick={() => setFiles([])}>Clear all</button>
                    </div>

                    <div className="preview-grid">
                        {files.map((file, idx) => (
                            <div key={`${file.name}-${idx}`} className="preview-card">
                                <img
                                    src={URL.createObjectURL(file)}
                                    alt={file.name}
                                    className="preview-img"
                                    onLoad={(e) => URL.revokeObjectURL(e.target.src)}
                                />
                                <button className="preview-remove" onClick={(e) => { e.stopPropagation(); removeFile(idx) }}>
                                    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                        <path d="M4.646 4.646a.5.5 0 0 1 .708 0L7 6.293l1.646-1.647a.5.5 0 0 1 .708.708L7.707 7l1.647 1.646a.5.5 0 0 1-.708.708L7 7.707 5.354 9.354a.5.5 0 0 1-.708-.708L6.293 7 4.646 5.354a.5.5 0 0 1 0-.708z" />
                                    </svg>
                                </button>
                                <div className="preview-name">{file.name}</div>
                            </div>
                        ))}
                    </div>

                    {/* Upload button */}
                    <div className="upload-actions">
                        {files.length < MIN_IMAGES && (
                            <p className="upload-warning">
                                ⚠ At least {MIN_IMAGES} images required ({MIN_IMAGES - files.length} more needed)
                            </p>
                        )}

                        {/* Segmentation toggle */}
                        <div className="segment-toggle-row">
                            <label className="toggle-switch" htmlFor="segment-toggle">
                                <input
                                    id="segment-toggle"
                                    type="checkbox"
                                    checked={segment}
                                    onChange={(e) => onSegmentChange(e.target.checked)}
                                />
                                <span className="toggle-slider" />
                            </label>
                            <div className="segment-label">
                                <span className="segment-label-text">Enable Segmentation</span>
                                <span className="segment-label-hint">
                                    Isolate the main object using AI before reconstruction
                                </span>
                            </div>
                        </div>

                        <button
                            className="btn btn-primary btn-lg"
                            disabled={files.length < MIN_IMAGES || uploading}
                            onClick={handleUpload}
                        >
                            {uploading ? (
                                <>
                                    <span className="spinner" />
                                    Uploading...
                                </>
                            ) : (
                                <>
                                    <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13a1 1 0 102 0V9.414l1.293 1.293a1 1 0 001.414-1.414z" clipRule="evenodd" />
                                    </svg>
                                    Start 3D Reconstruction
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Info cards */}
            <div className="info-cards">
                <div className="info-card">
                    <div className="info-icon">📸</div>
                    <h3>Multiple Angles</h3>
                    <p>Take photos from different positions around your scene for best results</p>
                </div>
                <div className="info-card">
                    <div className="info-icon">🔄</div>
                    <h3>Overlap is Key</h3>
                    <p>Ensure ~60-80% overlap between consecutive images for accurate matching</p>
                </div>
                <div className="info-card">
                    <div className="info-icon">✨</div>
                    <h3>AI-Powered</h3>
                    <p>Uses DUSt3R stereo matching inspired by the IMC 2025 winning solution</p>
                </div>
            </div>
        </div>
    )
}

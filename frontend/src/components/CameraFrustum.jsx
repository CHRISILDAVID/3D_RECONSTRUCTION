import { useMemo, useRef } from 'react'
import * as THREE from 'three'

/**
 * Renders a camera frustum wireframe at the given pose.
 * The frustum is a small pyramid showing the camera's FOV direction.
 */
export default function CameraFrustum({ pose, focal = 300, color = '#4ade80', size = 0.3 }) {
    const meshRef = useRef()

    const geometry = useMemo(() => {
        // Build a small frustum (pyramid) shape
        const s = size
        const d = s * 1.5 // depth

        const vertices = new Float32Array([
            // Apex (camera center)
            0, 0, 0,
            // Near plane corners
            -s, -s * 0.75, d,
            s, -s * 0.75, d,
            s, s * 0.75, d,
            -s, s * 0.75, d,
        ])

        const indices = [
            // Edges from apex to corners
            0, 1, 0, 2, 0, 3, 0, 4,
            // Near plane rectangle
            1, 2, 2, 3, 3, 4, 4, 1,
            // Top edge (image plane indicator)
            3, 4,
        ]

        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
        geo.setIndex(indices)

        return geo
    }, [size])

    // Build transform matrix from the 4x4 pose
    const matrix = useMemo(() => {
        if (!pose || pose.length !== 4) return new THREE.Matrix4()

        const m = new THREE.Matrix4()
        m.set(
            pose[0][0], pose[0][1], pose[0][2], pose[0][3],
            pose[1][0], pose[1][1], pose[1][2], pose[1][3],
            pose[2][0], pose[2][1], pose[2][2], pose[2][3],
            pose[3][0], pose[3][1], pose[3][2], pose[3][3],
        )

        // Invert because pose is world-from-camera
        const inv = m.clone().invert()
        return inv
    }, [pose])

    return (
        <lineSegments ref={meshRef} geometry={geometry} matrixAutoUpdate={false} matrix={matrix}>
            <lineBasicMaterial color={color} linewidth={2} transparent opacity={0.8} />
        </lineSegments>
    )
}

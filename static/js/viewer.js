'use strict';
(() => {
    const canvas = document.querySelector('#point-viewer');
    const status = document.querySelector('#viewer-status');
    const tooltip = document.querySelector('#point-tooltip');
    const gl = canvas.getContext('webgl', {
        antialias: true,
        alpha: true
    });
    if (!gl) {
        status.textContent = '3D is unavailable in this browser. See the static example below.';
        return;
    }
    const shader = (type, source) => {
        const result = gl.createShader(type);
        gl.shaderSource(result, source);
        gl.compileShader(result);
        if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(result));
        return result;
    };
    const program = gl.createProgram();
    gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec3 position; attribute vec3 color;
    uniform float yaw, pitch, zoom, aspect, pointSize, height;
    varying vec3 pointColor;
    void main() {
      float x = cos(yaw) * position.x - sin(yaw) * position.y;
      float depth = sin(yaw) * position.x + cos(yaw) * position.y;
      float y = cos(pitch) * position.z - sin(pitch) * depth;
      float z = sin(pitch) * position.z + cos(pitch) * depth;
      gl_Position = vec4(x * zoom / aspect, y * zoom, -z * 0.5, 3.5 - z);
      if (position.z > height) gl_Position = vec4(2., 2., 2., 1.);
      gl_PointSize = pointSize;
      pointColor = color;
    }`));
    gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float; varying vec3 pointColor;
    void main() {
      vec2 p = gl_PointCoord - vec2(0.5);
      if (dot(p, p) > 0.25) discard;
      gl_FragColor = vec4(pointColor, 1.0);
    }`));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    gl.enable(gl.DEPTH_TEST);
    const positionsBuffer = gl.createBuffer(),
        colorsBuffer = gl.createBuffer();
    const positionLocation = gl.getAttribLocation(program, 'position');
    const colorLocation = gl.getAttribLocation(program, 'color');
    const uniforms = Object.fromEntries(['yaw', 'pitch', 'zoom', 'aspect', 'pointSize', 'height'].map(name => [name, gl.getUniformLocation(program, name)]));
    const titles = {
        rgb: 'Original RGB',
        gt: 'Ground truth',
        prediction: 'PoVo prediction'
    };
    let mode = 'prediction',
        coloring = 'semantic',
        yaw = -2.1,
        pitch = 0.65,
        zoom = 2.6;
    let metadata, positions, rgb, ground, groundClasses, prediction, minHeight = -1,
        maxHeight = 1;
    let requestVersion = 0,
        pendingDraw = false,
        dragging = false,
        previousX, previousY;
    let displayedHeight = 85;
    const palette = identity => {
        if (identity < 0) return [0.39, 0.46, 0.43];
        const hue = (identity * 0.61803398875) % 1;
        const chroma = 0.57,
            value = 0.97;
        const x = chroma * (1 - Math.abs((hue * 6) % 2 - 1));
        const parts = [
            [chroma, x, 0],
            [x, chroma, 0],
            [0, chroma, x],
            [0, x, chroma],
            [x, 0, chroma],
            [chroma, 0, x]
        ][Math.floor(hue * 6)];
        return parts.map(channel => channel + value - chroma);
    };
    const pointInfo = index => {
        if (mode === 'rgb') return {
            label: 'Original RGB',
            classId: -1,
            identity: -1
        };
        if (mode === 'gt') return {
            label: groundClasses[index] >= 0 ? metadata.class_names[groundClasses[index]] : 'background / ignored',
            classId: groundClasses[index],
            identity: ground[index] - 1
        };
        const id = prediction[index];
        const item = metadata.prediction[id - 1];
        return item ? {
            label: item.label,
            classId: item.class_id,
            identity: id - 1,
            confidence: item.confidence
        } : {
            label: 'unassigned',
            classId: -1,
            identity: -1
        };
    };
    const draw = () => {
        pendingDraw = false;
        if (!positions) return;
        const bounds = canvas.getBoundingClientRect(),
            ratio = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.round(bounds.width * ratio),
            height = Math.round(bounds.height * ratio);
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        gl.viewport(0, 0, width, height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        Object.entries({
            yaw,
            pitch,
            zoom,
            aspect: width / height,
            pointSize: 2.6 * ratio,
            height: minHeight + (maxHeight - minHeight) * displayedHeight / 100
        }).forEach(([name, value]) => gl.uniform1f(uniforms[name], value));
        gl.bindBuffer(gl.ARRAY_BUFFER, positionsBuffer);
        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(positionLocation);
        gl.bindBuffer(gl.ARRAY_BUFFER, colorsBuffer);
        gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(colorLocation);
        gl.drawArrays(gl.POINTS, 0, metadata.displayed_points);
    };
    const scheduleDraw = () => {
        if (!pendingDraw) {
            pendingDraw = true;
            requestAnimationFrame(draw);
        }
    };
    const updateColors = () => {
        if (!metadata) return;
        const colors = new Float32Array(positions.length),
            classes = new Map();
        for (let i = 0; i < metadata.displayed_points; i++) {
            const info = pointInfo(i);
            const color = mode === 'rgb' ? rgb.subarray(i * 3, i * 3 + 3) : palette(info.classId < 0 ? -1 : coloring === 'semantic' ? info.classId : info.identity);
            colors.set(color, i * 3);
            if (mode !== 'rgb' && info.classId >= 0) classes.set(info.classId, info.label);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, colorsBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
        document.querySelector('#mode-title').textContent = titles[mode];
        const count = mode === 'gt' ? metadata.gt_instances : mode === 'rgb' ? null : metadata[mode].length;
        document.querySelector('#scene-details').textContent = metadata.point_count.toLocaleString() + ' original points · ' + metadata.displayed_points.toLocaleString() + ' displayed' + (count === null ? '' : ' · ' + count + ' foreground instances');
        const legend = document.querySelector('#class-legend');
        legend.replaceChildren();
        if (coloring === 'semantic' && mode !== 'rgb') {
            for (const [id, label] of Array.from(classes).sort((a, b) => a[1].localeCompare(b[1]))) {
                const chip = document.createElement('span'),
                    swatch = document.createElement('i');
                swatch.style.background = 'rgb(' + palette(id).map(v => Math.round(v * 255)).join(',') + ')';
                chip.append(swatch, document.createTextNode(label));
                legend.append(chip);
            }
        } else if (mode !== 'rgb') legend.textContent = 'Each foreground instance has a separate color. Switch to Class for a shared category palette.';
        tooltip.hidden = true;
        scheduleDraw();
    };
    async function loadScene(scene) {
        const version = ++requestVersion;
        status.textContent = 'Loading ' + scene + '…';
        try {
            const [infoResponse, binaryResponse] = await Promise.all([fetch('static/examples/' + scene + '/scene.json'), fetch('static/examples/' + scene + '/points.bin')]);
            if (!infoResponse.ok || !binaryResponse.ok) throw new Error('Could not fetch the example');
            const [info, buffer] = await Promise.all([infoResponse.json(), binaryResponse.arrayBuffer()]);
            if (version !== requestVersion) return;
            if (info.stride !== 22 || buffer.byteLength !== info.displayed_points * info.stride) throw new Error('Invalid point-cloud asset');
            metadata = info;
            const view = new DataView(buffer),
                count = info.displayed_points;
            positions = new Float32Array(count * 3);
            rgb = new Float32Array(count * 3);
            ground = new Int16Array(count);
            groundClasses = new Int16Array(count);
            prediction = new Uint16Array(count);
            minHeight = Infinity;
            maxHeight = -Infinity;
            for (let i = 0; i < count; i++) {
                const offset = i * info.stride;
                for (let axis = 0; axis < 3; axis++) {
                    positions[i * 3 + axis] = view.getFloat32(offset + axis * 4, true);
                    rgb[i * 3 + axis] = view.getUint8(offset + 12 + axis) / 255;
                }
                ground[i] = view.getInt16(offset + 16, true);
                groundClasses[i] = view.getInt16(offset + 18, true);
                prediction[i] = view.getUint16(offset + 20, true);
                minHeight = Math.min(minHeight, positions[i * 3 + 2]);
                maxHeight = Math.max(maxHeight, positions[i * 3 + 2]);
            }
            gl.bindBuffer(gl.ARRAY_BUFFER, positionsBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
            document.querySelector('#scene-title').textContent = scene;
            const viewCount = metadata.prediction_frames;
            status.textContent = `${viewCount ? viewCount + ' input views · ' : ''}fixed masks · real inference`;
            const accuracy = metadata.localized_class_accuracy;
            document.querySelector('#scene-accuracy').textContent = accuracy && accuracy.total > 0 ?
                `${(accuracy.accuracy * 100).toFixed(2)}% class accuracy · ${accuracy.correct}/${accuracy.total} localized instances · IoU ≥ 0.5` :
                'Class accuracy unavailable for this example.';
            updateColors();
        } catch (error) {
            if (version === requestVersion) status.textContent = 'Unable to load 3D. See the static example below.';
            console.error(error);
        }
    }
    document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
        mode = button.dataset.mode;
        document.querySelectorAll('[data-mode]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
        updateColors();
    }));
    document.querySelector('#color-select').addEventListener('change', e => {
        coloring = e.target.value;
        updateColors();
    });
    document.querySelector('#height-range').addEventListener('input', e => {
        displayedHeight = Number(e.target.value);
        document.querySelector('#height-value').textContent = displayedHeight + '%';
        tooltip.hidden = true;
        scheduleDraw();
    });
    const reset = () => {
        yaw = -2.1;
        pitch = .65;
        zoom = 2.6;
        displayedHeight = 85;
        document.querySelector("#height-range").value = 85;
        document.querySelector("#height-value").textContent = "85%";
        tooltip.hidden = true;
        scheduleDraw();
    };
    document.querySelector('#reset-view').addEventListener('click', reset);
    canvas.addEventListener('pointerdown', e => {
        dragging = true;
        previousX = e.clientX;
        previousY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
        tooltip.hidden = true;
    });
    canvas.addEventListener('pointerup', () => {
        dragging = false;
    });
    canvas.addEventListener('pointercancel', () => {
        dragging = false;
    });
    canvas.addEventListener('pointermove', e => {
        if (dragging) {
            yaw += (e.clientX - previousX) * .008;
            pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - previousY) * .008));
            previousX = e.clientX;
            previousY = e.clientY;
            scheduleDraw();
            return;
        }
        if (!positions || e.pointerType === 'touch') return;
        const bounds = canvas.getBoundingClientRect(),
            px = e.clientX - bounds.left,
            py = e.clientY - bounds.top,
            aspect = bounds.width / bounds.height;
        const cy = Math.cos(yaw),
            sy = Math.sin(yaw),
            cp = Math.cos(pitch),
            sp = Math.sin(pitch),
            height = minHeight + (maxHeight - minHeight) * displayedHeight / 100;
        let best = -1,
            bestDistance = 64,
            nearestDepth = -Infinity;
        for (let i = 0; i < metadata.displayed_points; i++) {
            const offset = i * 3,
                vx = positions[offset],
                vy = positions[offset + 1],
                vz = positions[offset + 2];
            if (vz > height) continue;
            const x = cy * vx - sy * vy,
                depth = sy * vx + cy * vy,
                y = cp * vz - sp * depth,
                z = sp * vz + cp * depth;
            const sx = (x * zoom / aspect / (3.5 - z) * .5 + .5) * bounds.width,
                syScreen = (.5 - y * zoom / (3.5 - z) * .5) * bounds.height;
            const dist = (sx - px) ** 2 + (syScreen - py) ** 2;
            if (dist < 64 && (dist < bestDistance - 5 || z > nearestDepth)) {
                best = i;
                bestDistance = dist;
                nearestDepth = z;
            }
        }
        tooltip.hidden = best < 0;
        if (best >= 0) {
            const info = pointInfo(best);
            tooltip.textContent = info.label + (info.identity >= 0 && mode !== 'rgb' ? ' · instance ' + (info.identity + 1) : '') + (info.confidence !== undefined ? ' · score ' + info.confidence.toFixed(3) : '');
            tooltip.style.left = Math.max(8, Math.min(px + 12, bounds.width - 235)) + 'px';
            tooltip.style.top = Math.max(8, py - 35) + 'px';
        }
    });
    canvas.addEventListener('pointerleave', () => {
        tooltip.hidden = true;
    });
    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        zoom = Math.max(1, Math.min(9, zoom * Math.exp(-e.deltaY * .001)));
        tooltip.hidden = true;
        scheduleDraw();
    }, {
        passive: false
    });
    canvas.addEventListener('keydown', e => {
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', '0'].includes(e.key)) return;
        e.preventDefault();
        if (e.key === 'ArrowLeft') yaw -= .1;
        if (e.key === 'ArrowRight') yaw += .1;
        if (e.key === 'ArrowUp') pitch = Math.min(1.45, pitch + .1);
        if (e.key === 'ArrowDown') pitch = Math.max(-1.45, pitch - .1);
        if (['+', '='].includes(e.key)) zoom = Math.min(9, zoom * 1.1);
        if (e.key === '-') zoom = Math.max(1, zoom / 1.1);
        if (e.key === '0') reset();
        scheduleDraw();
    });
    new ResizeObserver(scheduleDraw).observe(canvas.parentElement);
    loadScene(document.querySelector('#scene-name').textContent.trim());
})();

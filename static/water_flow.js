/**
 * Water Flow & Liquid Ripple Canvas Engine
 * High-performance fluid wave, caustic reflection, and interactive ripple simulation
 */

(function () {
    // Create or find canvas
    let canvas = document.getElementById('waterFlowCanvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'waterFlowCanvas';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '0';
        document.body.prepend(canvas);
    }

    const ctx = canvas.getContext('2d');
    let width = 0;
    let height = 0;
    let time = 0;

    // Interactive ripples
    const ripples = [];
    const maxRipples = 12;

    // Ambient floating bubbles / droplets
    const droplets = [];
    const numDroplets = 24;

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        initDroplets();
    }

    function initDroplets() {
        droplets.length = 0;
        for (let i = 0; i < numDroplets; i++) {
            droplets.push({
                x: Math.random() * width,
                y: Math.random() * height,
                radius: 2 + Math.random() * 5,
                speedY: 0.3 + Math.random() * 0.7,
                speedX: (Math.random() - 0.5) * 0.4,
                opacity: 0.2 + Math.random() * 0.35,
                wobble: Math.random() * Math.PI * 2
            });
        }
    }

    window.addEventListener('resize', resize);
    resize();

    // Mouse interactive ripples
    window.addEventListener('mousemove', (e) => {
        if (Math.random() < 0.25) {
            addRipple(e.clientX, e.clientY);
        }
    });

    window.addEventListener('click', (e) => {
        addRipple(e.clientX, e.clientY, 1.6);
        addRipple(e.clientX, e.clientY, 1.0);
    });

    window.addEventListener('touchmove', (e) => {
        if (e.touches.length > 0 && Math.random() < 0.3) {
            addRipple(e.touches[0].clientX, e.touches[0].clientY);
        }
    }, { passive: true });

    function addRipple(x, y, intensity = 1.0) {
        if (ripples.length >= maxRipples) {
            ripples.shift();
        }
        ripples.push({
            x,
            y,
            radius: 4,
            maxRadius: 100 * intensity,
            speed: 1.8 * intensity,
            opacity: 0.45 * intensity
        });
    }

    // Fluid Wave Layers configuration
    const waveLayers = [
        { amp: 28, freq: 0.0035, speed: 0.022, yOffset: 0.25, color: 'rgba(255, 238, 195, 0.14)' },
        { amp: 35, freq: 0.0028, speed: -0.018, yOffset: 0.45, color: 'rgba(235, 195, 120, 0.12)' },
        { amp: 42, freq: 0.0042, speed: 0.028, yOffset: 0.68, color: 'rgba(255, 245, 215, 0.16)' },
        { amp: 30, freq: 0.0030, speed: -0.015, yOffset: 0.85, color: 'rgba(215, 175, 95, 0.13)' }
    ];

    function drawWave(layer, t) {
        const baseY = height * layer.yOffset;
        ctx.beginPath();
        ctx.moveTo(0, height);
        ctx.lineTo(0, baseY);

        for (let x = 0; x <= width; x += 15) {
            const y1 = Math.sin(x * layer.freq + t * layer.speed) * layer.amp;
            const y2 = Math.cos(x * layer.freq * 0.7 - t * layer.speed * 1.3) * (layer.amp * 0.5);
            ctx.lineTo(x, baseY + y1 + y2);
        }

        ctx.lineTo(width, height);
        ctx.closePath();
        ctx.fillStyle = layer.color;
        ctx.fill();

        // Shimmering crest outline
        ctx.beginPath();
        for (let x = 0; x <= width; x += 15) {
            const y1 = Math.sin(x * layer.freq + t * layer.speed) * layer.amp;
            const y2 = Math.cos(x * layer.freq * 0.7 - t * layer.speed * 1.3) * (layer.amp * 0.5);
            if (x === 0) ctx.moveTo(x, baseY + y1 + y2);
            else ctx.lineTo(x, baseY + y1 + y2);
        }
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    // Dynamic Caustic Refraction Streams
    function drawCaustics(t) {
        ctx.save();
        ctx.lineWidth = 2.0;

        const count = 6;
        for (let i = 0; i < count; i++) {
            const progress = (i / count) + (t * 0.003);
            const yStart = (progress % 1.0) * height;

            ctx.beginPath();
            ctx.strokeStyle = `rgba(255, 248, 225, ${0.08 + Math.sin(t * 0.05 + i) * 0.04})`;

            for (let x = 0; x <= width; x += 25) {
                const wave1 = Math.sin(x * 0.005 + t * 0.03 + i * 1.5) * 20;
                const wave2 = Math.cos(x * 0.002 - t * 0.02 + i) * 15;
                const cy = yStart + wave1 + wave2;

                if (x === 0) ctx.moveTo(x, cy);
                else ctx.lineTo(x, cy);
            }
            ctx.stroke();
        }
        ctx.restore();
    }

    // Floating water bubbles
    function updateAndDrawDroplets(t) {
        ctx.save();
        for (let i = 0; i < droplets.length; i++) {
            const d = droplets[i];
            d.y -= d.speedY;
            d.wobble += 0.03;
            d.x += Math.sin(d.wobble) * 0.6 + d.speedX;

            if (d.y < -10) {
                d.y = height + 10;
                d.x = Math.random() * width;
            }

            // Draw bubble with refraction highlight
            ctx.beginPath();
            ctx.arc(d.x, d.y, d.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 250, 230, ${d.opacity * 0.35})`;
            ctx.fill();

            // Specular glint
            ctx.beginPath();
            ctx.arc(d.x - d.radius * 0.3, d.y - d.radius * 0.3, d.radius * 0.35, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${d.opacity * 0.7})`;
            ctx.fill();
        }
        ctx.restore();
    }

    // Interactive circular ripples
    function updateAndDrawRipples() {
        for (let i = ripples.length - 1; i >= 0; i--) {
            const r = ripples[i];
            r.radius += r.speed;
            r.opacity *= 0.965;

            if (r.radius > r.maxRadius || r.opacity < 0.01) {
                ripples.splice(i, 1);
                continue;
            }

            // Outer ring
            ctx.beginPath();
            ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(255, 255, 255, ${r.opacity})`;
            ctx.lineWidth = 1.8;
            ctx.stroke();

            // Inner echo ring
            if (r.radius > 15) {
                ctx.beginPath();
                ctx.arc(r.x, r.y, r.radius - 10, 0, Math.PI * 2);
                ctx.strokeStyle = `rgba(235, 195, 120, ${r.opacity * 0.5})`;
                ctx.lineWidth = 1.0;
                ctx.stroke();
            }
        }
    }

    // Main animation loop
    function animate() {
        ctx.clearRect(0, 0, width, height);
        time += 1;

        // 1. Fluid waves
        for (let i = 0; i < waveLayers.length; i++) {
            drawWave(waveLayers[i], time);
        }

        // 2. Caustic streams
        drawCaustics(time);

        // 3. Floating bubbles
        updateAndDrawDroplets(time);

        // 4. Interactive user ripples
        updateAndDrawRipples();

        requestAnimationFrame(animate);
    }

    animate();
})();

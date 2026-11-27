(() => {
    const stream = document.getElementById('demo-stream');
    const prompt = document.getElementById('demo-prompt');
    const status = document.getElementById('demo-status');
    const haunt = document.getElementById('demo-haunt');
    const sensitivity = document.getElementById('demo-sensitivity');
    const sensitivityValue = document.getElementById('demo-sensitivity-value');
    const rule = document.getElementById('demo-rule');
    const trail = document.getElementById('demo-trail');
    const stepSlider = document.getElementById('demo-step-slider');
    const selectedStep = document.getElementById('demo-step');
    const selectedToken = document.getElementById('demo-token');
    const selectedConfidence = document.getElementById('demo-confidence');
    const selectedEntropy = document.getElementById('demo-entropy');
    const selectedMargin = document.getElementById('demo-margin');
    const alternativesNode = document.getElementById('demo-alternatives');
    let examples = {};
    let steps = [];
    let promptTokens = [];
    let selectedIndex = -1;

    function currentThresholds() {
        return GhostSignals.thresholdsForSensitivity(sensitivity.value);
    }

    function tokenLabel(token) {
        return (token || '').replace(/\n/g, '↵').replace(/\r/g, '␍').replace(/ /g, '␠') || 'empty token';
    }

    function alternativesFor(step) {
        const ids = step.top_logit_ids || [];
        const values = step.top_logit_values || [];
        const strings = step.top_logit_strs || [];
        if (!values.length) return [];
        const maxLogit = Math.max(...values);
        const exponentials = values.map(value => Math.exp(value - maxLogit));
        const denominator = exponentials.reduce((sum, value) => sum + value, 0);
        return ids.map((id, index) => ({
            id: Number(id),
            token: strings[index] || `[${id}]`,
            probability: exponentials[index] / denominator,
            logit: values[index],
        }))
            .filter(item => item.id !== Number(step.token_id))
            .sort((left, right) => right.logit - left.logit)
            .slice(0, 3);
    }

    function updateRule() {
        const thresholds = currentThresholds();
        sensitivityValue.textContent = `${sensitivity.value}%`;
        rule.textContent = `Ghost rule: H ≥ ${thresholds.entropy.toFixed(1)} bits AND raw P < ${(thresholds.probability * 100).toFixed(1)}%.`;
    }

    function renderHaunt() {
        if (!steps.length) {
            haunt.textContent = 'haunt: —';
            haunt.dataset.level = 'idle';
            return;
        }
        const countable = steps.filter(step => GhostSignals.isCountableToken(step.token_str));
        if (!countable.length) {
            haunt.textContent = 'haunt: — · 0 content tokens';
            haunt.dataset.level = 'idle';
            return;
        }
        const ghosts = countable.filter(step => GhostSignals.isGhost(step, sensitivity.value)).length;
        const percent = GhostSignals.hauntPercent(steps, sensitivity.value);
        haunt.textContent = `haunt: ${percent}% · ${ghosts}/${countable.length} content tokens`;
        haunt.dataset.level = percent < 30 ? 'low' : percent < 60 ? 'medium' : 'high';
    }

    function renderStream() {
        stream.replaceChildren();
        promptTokens.forEach(token => {
            const chip = document.createElement('span');
            chip.className = 'chip chip-prompt';
            chip.textContent = token;
            stream.appendChild(chip);
        });
        steps.forEach((step, index) => {
            const wrapper = document.createElement('span');
            wrapper.className = 'chip-wrap';
            const chip = document.createElement('span');
            const ghost = GhostSignals.isGhost(step, sensitivity.value);
            chip.className = `chip chip-gen ${ghost ? 'chip-ghost' : 'chip-solid'}${index === selectedIndex ? ' chip-active' : ''}`;
            chip.dataset.step = index;
            chip.textContent = step.token_str || '';
            chip.title = tooltipForStep(step);
            wrapper.appendChild(chip);
            stream.appendChild(wrapper);
        });
        renderHaunt();
        renderTrail();
    }

    function tooltipForStep(step) {
        const topAlternatives = alternativesFor(step)
            .map(item => `  ${tokenLabel(item.token)} (${(item.probability * 100).toFixed(2)}% of top 200)`);
        return [
            `Raw confidence (full vocabulary): ${(Number(step.p_chosen) * 100).toFixed(3)}%`,
            `Entropy: ${Number(step.entropy_bits).toFixed(3)} bits`,
            `Margin: ${(Number(step.margin) * 100).toFixed(3)} percentage points`,
            'Top alternatives:',
            ...topAlternatives,
        ].join('\n');
    }

    function renderTrail() {
        const context = trail.getContext('2d');
        const width = Math.max(1, trail.clientWidth);
        const height = Math.max(1, trail.clientHeight);
        const ratio = window.devicePixelRatio || 1;
        trail.width = Math.round(width * ratio);
        trail.height = Math.round(height * ratio);
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);
        const left = 38, right = 142, top = 8, bottom = 22;
        const plotWidth = Math.max(1, width - left - right);
        const plotHeight = Math.max(1, height - top - bottom);
        const yFor = entropy => top + plotHeight * (1 - entropy / 16);
        const xFor = index => left + (steps.length <= 1 ? plotWidth / 2 : index * plotWidth / (steps.length - 1));
        context.font = '12px "JetBrains Mono", monospace';
        context.fillStyle = '#b9a8d6';
        context.textAlign = 'left';
        context.fillText('16 bits', 2, top + 4);
        context.fillText('0 bits', 2, top + plotHeight + 3);
        const thresholdY = yFor(currentThresholds().entropy);
        context.setLineDash([4, 3]);
        context.strokeStyle = 'rgba(181, 140, 255, 0.8)';
        context.beginPath();
        context.moveTo(left, thresholdY);
        context.lineTo(left + plotWidth, thresholdY);
        context.stroke();
        context.setLineDash([]);
        const label = `${currentThresholds().entropy.toFixed(1)} bits threshold`;
        const labelWidth = context.measureText(label).width;
        context.fillStyle = '#090a0f';
        context.fillRect(width - right + 4, thresholdY - 13, labelWidth + 4, 16);
        context.fillStyle = '#decaff';
        context.textAlign = 'left';
        context.fillText(label, width - right + 6, Math.max(top + 14, thresholdY - 3));
        if (!steps.length) return;
        context.beginPath();
        context.strokeStyle = '#b58cff';
        context.lineWidth = 1.5;
        steps.forEach((step, index) => {
            const x = xFor(index);
            const y = yFor(Number(step.entropy_bits) || 0);
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        });
        context.stroke();
        steps.forEach((step, index) => {
            context.beginPath();
            context.arc(xFor(index), yFor(Number(step.entropy_bits) || 0), index === selectedIndex ? 4 : 3, 0, Math.PI * 2);
            context.fillStyle = GhostSignals.isGhost(step, sensitivity.value) ? '#6fe7b7' : '#cbb4ff';
            context.fill();
        });
    }

    function selectStep(index) {
        if (index < 0 || index >= steps.length) return;
        selectedIndex = index;
        stepSlider.value = index;
        const step = steps[index];
        selectedStep.textContent = `${index + 1} / ${steps.length}`;
        selectedToken.textContent = tokenLabel(step.token_str);
        selectedConfidence.textContent = `${(Number(step.p_chosen) * 100).toFixed(2)}%`;
        selectedEntropy.textContent = `${Number(step.entropy_bits).toFixed(2)} bits`;
        selectedMargin.textContent = `${(Number(step.margin) * 100).toFixed(2)} pp`;
        alternativesNode.replaceChildren();
        alternativesFor(step).forEach(item => {
            const node = document.createElement('span');
            node.className = 'selected-alternative';
            node.textContent = `${tokenLabel(item.token)} · ${(item.probability * 100).toFixed(1)}%`;
            alternativesNode.appendChild(node);
        });
        renderStream();
    }

    function showExample(id) {
        const example = examples[id];
        if (!example) return;
        prompt.textContent = example.prompt;
        promptTokens = example.prompt_token_strs || [];
        steps = example.steps || [];
        selectedIndex = -1;
        stepSlider.max = Math.max(0, steps.length - 1);
        renderStream();
        if (steps.length) {
            let highest = 0;
            steps.forEach((step, index) => {
                if (Number(step.entropy_bits) > Number(steps[highest].entropy_bits)) highest = index;
            });
            selectStep(highest);
        }
        status.textContent = `${example.label} · saved trace`;
    }

    sensitivity.addEventListener('input', () => {
        updateRule();
        renderStream();
        if (selectedIndex >= 0) selectStep(selectedIndex);
    });
    stepSlider.addEventListener('input', () => selectStep(Number(stepSlider.value)));
    stream.addEventListener('click', event => {
        const chip = event.target.closest('[data-step]');
        if (chip) selectStep(Number(chip.dataset.step));
    });
    trail.addEventListener('click', event => {
        if (!steps.length) return;
        const bounds = trail.getBoundingClientRect();
        const left = 32, right = 8;
        const width = Math.max(1, bounds.width - left - right);
        const x = event.clientX - bounds.left - left;
        const index = steps.length <= 1 ? 0 : Math.max(0, Math.min(steps.length - 1, Math.round(x * (steps.length - 1) / width)));
        selectStep(index);
    });
    document.querySelectorAll('[data-example]').forEach(button => {
        button.addEventListener('click', () => showExample(button.dataset.example));
    });
    window.addEventListener('resize', renderTrail);

    updateRule();
    fetch('demo-trace.json')
        .then(response => {
            if (!response.ok) throw new Error(`Trace request failed (${response.status}).`);
            return response.json();
        })
        .then(trace => {
            if (trace.version !== 1 || !trace.examples || !trace.examples[trace.default]) {
                throw new Error('Unsupported saved trace format.');
            }
            examples = trace.examples;
            showExample(trace.default);
        })
        .catch(error => {
            status.className = 'demo-status demo-error';
            status.textContent = 'saved trace unavailable';
            console.error('Unable to load static demo trace:', error);
        });
})();

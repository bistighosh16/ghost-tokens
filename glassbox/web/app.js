/* Ghost Tokens App Controller */

(() => {
    const generateBtn = document.getElementById('generate-btn');
    const promptInput = document.getElementById('prompt');
    const maxTokensInput = document.getElementById('max-tokens');
    const useCacheInput = document.getElementById('use-cache');
    const tokenStream = document.getElementById('token-stream');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const metricSpeed = document.getElementById('metric-speed');
    const metricCount = document.getElementById('metric-count');
    const metricCache = document.getElementById('metric-cache');
    const attnLayerSel = document.getElementById('attn-layer');
    const attnHeadSel = document.getElementById('attn-head');
    const attnAvgHeads = document.getElementById('attn-avg-heads');
    const attnRowNorm = document.getElementById('attn-row-norm');
    const attnHideSink = document.getElementById('attn-hide-sink');
    const attnStepLabel = document.getElementById('attn-step-label');
    const underhood = document.querySelector('.panel-underhood');
    const stepSlider = document.getElementById('step-slider');
    const stepIndicator = document.getElementById('step-indicator');
    const selectedToken = document.getElementById('selected-token');
    const selectedConfidence = document.getElementById('selected-confidence');
    const selectedEntropy = document.getElementById('selected-entropy');
    const selectedMargin = document.getElementById('selected-margin');
    const selectedAlternatives = document.getElementById('selected-alternatives');
    const emptyHint = document.getElementById('empty-hint');
    const traceSourceLabel = document.getElementById('trace-source-label');
    const hauntBadge = document.getElementById('haunt-badge');
    const ghostThreshold = document.getElementById('ghost-threshold');
    const ghostThresholdValue = document.getElementById('ghost-threshold-value');
    const ghostRule = document.getElementById('ghost-rule');
    const confidenceTrail = document.getElementById('confidence-trail');
    const replayPlay = document.getElementById('replay-play');
    const replayStep = document.getElementById('replay-step');
    const replayStop = document.getElementById('replay-stop');
    const replayAlternatives = document.getElementById('replay-alternatives');
    const spookyMode = document.getElementById('spooky-mode');
    const branchBanner = document.getElementById('branch-banner');
    const branchLabel = document.getElementById('branch-label');
    const backToOriginalBtn = document.getElementById('back-to-original');
    const headGallery = document.getElementById('head-gallery');
    const headGalleryGrid = document.getElementById('head-gallery-grid');
    const headGalleryLayer = document.getElementById('head-gallery-layer');

    let steps = [];
    let config = null;
    let promptTokens = [];
    let genTokens = [];
    let allAttentionData = [];
    let ws = null;
    let selectedStepIdx = -1;
    let originalPrompt = '';
    let originalMaxTokens = maxTokensInput.value;
    let runPrompt = '';
    let replayTimer = null;
    let replayMode = false;
    let generationActive = false;
    let branchComparison = null;
    let demoExamples = null;

    function ghostThresholds() {
        return GhostSignals.thresholdsForSensitivity(ghostThreshold.value);
    }

    function setStatus(state, text) {
        statusDot.className = `dot dot-${state}`;
        statusText.textContent = text;
    }

    function renderTokenStream(activeIdx = -1) {
        let html = '';
        promptTokens.forEach(t => {
            html += `<span class="chip chip-prompt">${vis(t || '')}</span>`;
        });
        steps.forEach((s, idx) => {
            const isActive = idx === activeIdx;
            const ghost = isGhost(s);
            const cls = isActive ? 'chip chip-gen chip-active' : 'chip chip-gen';
            const details = tooltipForStep(s);
            const exorcise = isActive && ghost
                ? `<button class="exorcise-btn" type="button" data-exorcise-step="${idx}" title="Force the best other candidate and continue">Exorcise</button>`
                : '';
            html += `<span class="chip-wrap"><span class="${cls} ${ghost ? 'chip-ghost' : 'chip-solid'}" data-step="${idx}" title="${escapeAttr(details)}">${vis(s.token_str || '')}</span>${exorcise}</span>`;
        });
        tokenStream.innerHTML = html;
        emptyHint.hidden = steps.length > 0;
        updateHauntBadge();
        renderConfidenceTrail();
        updateReplayAlternatives();
    }

    function isGhost(step) {
        return GhostSignals.isGhost(step, ghostThreshold.value);
    }

    function hauntPercent(items) {
        return GhostSignals.hauntPercent(items, ghostThreshold.value);
    }

    function updateHauntBadge() {
        if (!steps.length) {
            hauntBadge.textContent = '';
            hauntBadge.dataset.level = 'idle';
            const fill = document.getElementById('haunt-meter-fill');
            if (fill) fill.style.width = '0%';
            const value = document.getElementById('haunt-meter-value');
            if (value) value.textContent = '—';
            const tier = document.getElementById('haunt-tier');
            if (tier) tier.textContent = 'Calm';
            return;
        }
        const countable = steps.filter(step => GhostSignals.isCountableToken(step.token_str));
        if (!countable.length) {
            hauntBadge.textContent = '';
            hauntBadge.dataset.level = 'idle';
            return;
        }
        const ghostCount = countable.filter(isGhost).length;
        const percent = hauntPercent(steps);
        hauntBadge.textContent = '';
        hauntBadge.dataset.level = percent < 30 ? 'low' : percent < 60 ? 'medium' : 'high';
        const tier = percent < 30 ? 'Calm' : percent < 60 ? 'Restless' : 'Haunted';
        const fill = document.getElementById('haunt-meter-fill');
        if (fill) fill.style.width = `${percent}%`;
        const value = document.getElementById('haunt-meter-value');
        if (value) value.textContent = `${percent}% · ${ghostCount} of ${countable.length} words`;
        const tierNode = document.getElementById('haunt-tier');
        if (tierNode) tierNode.textContent = tier;
    }

    function getAlternatives(step, limit = 5) {
        const ids = step.top_logit_ids || [];
        const values = step.top_logit_values || [];
        const strings = step.top_logit_strs || [];
        if (!values.length) return [];
        const maxLogit = Math.max(...values);
        const expValues = values.map(value => Math.exp(value - maxLogit));
        const denominator = expValues.reduce((sum, value) => sum + value, 0);
        return ids.map((id, index) => ({
            id: Number(id),
            token: strings[index] || `[${id}]`,
            probability: expValues[index] / denominator,
            logit: values[index],
        }))
            .filter(item => item.id !== Number(step.token_id))
            .sort((a, b) => b.logit - a.logit)
            .slice(0, limit);
    }

    function tooltipForStep(step) {
        const probability = Number(step.p_chosen);
        const alternatives = getAlternatives(step, 5)
            .map(item => `  ${item.token.replace(/\s+/g, ' ') || '∅'} (${(item.probability * 100).toFixed(2)}% of top 200)`);
        return [
            `Raw confidence (full vocabulary): ${Number.isFinite(probability) ? (probability * 100).toFixed(3) + '%' : 'unavailable'}`,
            `Entropy: ${Number(step.entropy_bits).toFixed(3)} bits`,
            `Margin: ${(Number(step.margin) * 100).toFixed(3)} percentage points`,
            'Top alternatives (normalized within top 200):',
            ...(alternatives.length ? alternatives : ['  none available']),
        ].join('\n');
    }

    function escapeAttr(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    function renderConfidenceTrail() {
        const context = confidenceTrail.getContext('2d');
        if (!context) return;
        const width = Math.max(1, confidenceTrail.clientWidth);
        const height = Math.max(1, confidenceTrail.clientHeight);
        const ratio = window.devicePixelRatio || 1;
        confidenceTrail.width = Math.round(width * ratio);
        confidenceTrail.height = Math.round(height * ratio);
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);

        const left = 38, right = 142, top = 8, bottom = 22;
        const plotWidth = Math.max(1, width - left - right);
        const plotHeight = Math.max(1, height - top - bottom);
        const maxEntropy = 16;
        const xFor = index => left + (steps.length <= 1 ? plotWidth / 2 : index * plotWidth / (steps.length - 1));
        const yFor = entropy => top + plotHeight * (1 - entropy / maxEntropy);

        context.font = '12px "JetBrains Mono", monospace';
        context.fillStyle = '#b9a8d6';
        context.textAlign = 'left';
        context.fillText(`${maxEntropy.toFixed(0)} bits`, 2, top + 4);
        context.fillText('0 bits', 2, top + plotHeight + 3);
        context.strokeStyle = 'rgba(180, 200, 240, 0.18)';
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(left, top + plotHeight);
        context.lineTo(left + plotWidth, top + plotHeight);
        context.stroke();
        if (steps.length) {
            context.textAlign = 'left';
            context.fillText('1', left, height - 3);
            context.textAlign = 'right';
            context.fillText(String(steps.length), left + plotWidth, height - 3);
        } else {
            context.textAlign = 'right';
            context.fillText('token position →', width - right, height - 3);
        }

        const thresholdY = yFor(ghostThresholds().entropy);
        if (thresholdY >= top && thresholdY <= top + plotHeight) {
            context.beginPath();
            context.setLineDash([4, 3]);
            context.strokeStyle = 'rgba(181, 140, 255, 0.8)';
            context.moveTo(left, thresholdY);
            context.lineTo(left + plotWidth, thresholdY);
            context.stroke();
            context.setLineDash([]);
            context.fillStyle = '#d5bdff';
            context.textAlign = 'right';
            const thresholdLabel = `${ghostThresholds().entropy.toFixed(1)} bits threshold`;
            const labelY = Math.max(top + 14, thresholdY - 3);
            const labelWidth = context.measureText(thresholdLabel).width;
            context.fillStyle = '#090a0f';
            context.fillRect(width - right + 4, labelY - 13, labelWidth + 4, 16);
            context.fillStyle = '#d5bdff';
            context.textAlign = 'left';
            context.fillText(thresholdLabel, width - right + 6, labelY);
        }

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
            const x = xFor(index);
            const y = yFor(Number(step.entropy_bits) || 0);
            context.beginPath();
            context.arc(x, y, isGhost(step) ? 3.5 : 2.2, 0, Math.PI * 2);
            context.fillStyle = isGhost(step) ? '#6fe7b7' : '#cbb4ff';
            context.fill();
        });
    }

    function populateSelectors() {
        attnLayerSel.innerHTML = ''; attnHeadSel.innerHTML = '';
        for (let i = 0; i < config.num_layers; i++) {
            const o = document.createElement('option');
            o.value = i; o.textContent = `L${i}`;
            attnLayerSel.appendChild(o);
        }
        for (let i = 0; i < config.num_heads; i++) {
            const o = document.createElement('option');
            o.value = i; o.textContent = `H${i}`;
            attnHeadSel.appendChild(o);
        }
        attnLayerSel.value = Math.floor(config.num_layers / 2);
    }

    function updateAttention() {
        if (!underhood.open) return;
        if (config && allAttentionData.length > 0 && selectedStepIdx >= 0) {
            const selectedToken = steps[selectedStepIdx]?.token_str || '';
            attnStepLabel.textContent = `Selected step ${selectedStepIdx + 1}/${steps.length}: ${readableToken(selectedToken)}`;
            Heatmap.render(
                allAttentionData.slice(0, selectedStepIdx + 1),
                promptTokens,
                genTokens.slice(0, selectedStepIdx + 1),
                parseInt(attnLayerSel.value),
                parseInt(attnHeadSel.value),
                attnAvgHeads.checked,
                attnRowNorm.checked,
                attnHideSink.checked
            );
        }
        updateAttentionHighlights();
        renderHeadGallery();
    }

    function updateAttentionHighlights() {
        const chips = Array.from(tokenStream.querySelectorAll('.chip'));
        chips.forEach(chip => {
            chip.classList.remove('chip-attention');
            chip.style.removeProperty('background-color');
        });
        if (selectedStepIdx < 0 || !steps[selectedStepIdx]) return;

        const step = steps[selectedStepIdx];
        const layerRows = step.attention_new_rows && step.attention_new_rows[parseInt(attnLayerSel.value)];
        if (!Array.isArray(layerRows) || layerRows.length === 0) return;

        let row;
        if (attnAvgHeads.checked) {
            const validRows = layerRows.filter(Array.isArray);
            if (!validRows.length) return;
            const rowLength = Math.min(...validRows.map(values => values.length));
            row = Array.from({ length: rowLength }, (_, index) =>
                validRows.reduce((sum, values) => sum + values[index], 0) / validRows.length
            );
        } else {
            row = layerRows[parseInt(attnHeadSel.value)];
        }
        if (!Array.isArray(row)) return;

        const chipsBeforeStep = chips.filter((chip) =>
            chip.classList.contains('chip-prompt') || Number(chip.dataset.step) < selectedStepIdx
        );
        if (row.length !== chipsBeforeStep.length) {
            console.warn(`Attention row length ${row.length} does not match ${chipsBeforeStep.length} chips before step ${selectedStepIdx + 1}`);
            return;
        }

        const weights = [...row];
        if (weights.length) weights[0] = 0;
        const maxWeight = Math.max(0, ...weights);
        if (maxWeight <= 0) return;
        chipsBeforeStep.forEach((chip, index) => {
            const alpha = Math.min(0.55, 0.55 * weights[index] / maxWeight);
            if (alpha <= 0) return;
            chip.classList.add('chip-attention');
            chip.style.backgroundColor = `rgba(181, 140, 255, ${alpha})`;
        });
    }

    function renderHeadGallery() {
        if (!headGallery.open || !config || !allAttentionData.length) return;
        const layerIdx = parseInt(attnLayerSel.value);
        headGalleryLayer.textContent = `L${layerIdx}`;
        headGalleryGrid.innerHTML = '';
        for (let headIdx = 0; headIdx < config.num_heads; headIdx++) {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'head-gallery-item';
            item.dataset.head = headIdx;
            item.setAttribute('aria-label', `Select head ${headIdx}`);

            const label = document.createElement('span');
            label.textContent = `H${headIdx}`;
            const canvas = document.createElement('canvas');
            canvas.width = 168;
            canvas.height = 72;
            canvas.setAttribute('aria-hidden', 'true');
            const matrix = allAttentionData.map(stepData =>
                stepData && stepData[layerIdx] && stepData[layerIdx][headIdx]
                    ? stepData[layerIdx][headIdx]
                    : []
            );
            Heatmap.drawMini(canvas, matrix, attnRowNorm.checked, attnHideSink.checked);
            item.append(label, canvas);
            item.addEventListener('click', () => {
                attnHeadSel.value = String(headIdx);
                attnAvgHeads.checked = false;
                attnHeadSel.dispatchEvent(new Event('change', { bubbles: true }));
                headGallery.close();
            });
            headGalleryGrid.appendChild(item);
        }
    }

    function selectStep(idx) {
        if (idx < 0 || idx >= steps.length) return;
        selectedStepIdx = idx;
        stepSlider.value = idx;
        const step = steps[idx];
        stepIndicator.textContent = `${idx + 1} / ${steps.length}`;
        selectedToken.textContent = readableToken(step.token_str);
        selectedConfidence.textContent = `${(Number(step.p_chosen) * 100).toFixed(2)}%`;
        selectedEntropy.textContent = `${Number(step.entropy_bits).toFixed(2)} bits`;
        selectedMargin.textContent = `${(Number(step.margin) * 100).toFixed(2)} pp`;
        renderSelectedAlternatives(step);

        ProbBars.setStep(step.top_logit_ids, step.top_logit_values, step.top_logit_strs, step.token_id);
        Lens.render(step.lens_predictions);
        renderTokenStream(idx);
        updateAttention();
    }

    function readableToken(token) {
        return (token || '').replace(/\n/g, '↵').replace(/\r/g, '␍').replace(/ /g, '␠') || 'empty token';
    }

    function renderSelectedAlternatives(step) {
        selectedAlternatives.replaceChildren();
        const alternatives = getAlternatives(step, 3);
        if (!alternatives.length) {
            selectedAlternatives.textContent = 'No alternatives available for this step.';
            return;
        }
        alternatives.forEach(item => {
            const alternative = document.createElement('span');
            alternative.className = 'selected-alternative';
            alternative.textContent = `${readableToken(item.token)} · ${(item.probability * 100).toFixed(1)}%`;
            selectedAlternatives.appendChild(alternative);
        });
    }

    function clearSelectedWord() {
        stepIndicator.textContent = '—';
        selectedToken.textContent = 'Select a word';
        selectedConfidence.textContent = '—';
        selectedEntropy.textContent = '—';
        selectedMargin.textContent = '—';
        selectedAlternatives.textContent = 'Select a word to inspect alternatives.';
    }

    function selectHighestEntropyStep() {
        if (steps.length === 0) return;
        let maxH = -1;
        let maxIdx = 0;
        steps.forEach((s, idx) => {
            const h = s.entropy_bits;
            if (h > maxH) {
                maxH = h;
                maxIdx = idx;
            }
        });
        stepSlider.value = maxIdx;
        selectStep(maxIdx);
    }

    function updateBranchComparison() {
        if (!branchComparison) return;
        branchLabel.textContent = `haunt before: ${branchComparison.before}% · after branch suffix: ${hauntPercent(steps)}%`;
    }

    function startGeneration(prompt, { branch = false, resetOriginal = true, comparison = null } = {}) {
        stopReplay();
        if (resetOriginal) {
            originalPrompt = prompt;
            originalMaxTokens = maxTokensInput.value;
        }
        runPrompt = prompt;
        promptInput.value = prompt;
        traceSourceLabel.textContent = 'Live generation · confidence updates as tokens arrive.';
        branchBanner.hidden = !branch;
        branchComparison = branch ? comparison : null;
        if (ws) ws.close();
        steps = []; genTokens = []; allAttentionData = [];
        selectedStepIdx = -1;
        tokenStream.innerHTML = '';
        metricSpeed.textContent = '—';
        metricCount.textContent = '—';
        metricCache.textContent = '—';
        clearSelectedWord();
        generationActive = true;
        renderTokenStream();
        
        setStatus('active', 'connecting…');
        generateBtn.disabled = true;

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = new WebSocket(`${protocol}//${location.host}/ws/generate`);
        ws = socket;

        socket.onopen = () => {
            setStatus('active', 'generating…');
            socket.send(JSON.stringify({
                prompt: runPrompt,
                max_new_tokens: parseInt(maxTokensInput.value),
                temperature: parseFloat(document.getElementById('temp-slider').value),
                top_k: parseInt(document.getElementById('topk-slider').value),
                top_p: parseFloat(document.getElementById('topp-slider').value),
                use_cache: useCacheInput.checked,
                trace: true,
                seed: 1337,
            }));
        };

        socket.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === 'init') {
                config = msg.config;
                promptTokens = msg.prompt_token_strs;
                populateSelectors();
                renderTokenStream();
            } else if (msg.type === 'step') {
                steps.push(msg);
                genTokens.push(msg.token_str);
                allAttentionData.push(msg.attention_new_rows);
                stepSlider.max = steps.length - 1;
                if (!replayMode) stepSlider.value = steps.length - 1;
                
                metricSpeed.textContent = (steps.reduce((a, s) => a + s.tokens_per_sec, 0) / steps.length).toFixed(1);
                metricCount.textContent = steps.filter(step => GhostSignals.isCountableToken(step.token_str)).length;
                metricCache.textContent = msg.cache_active ? 'on' : 'off';
                
                renderTokenStream(replayMode ? selectedStepIdx : -1);
                updateBranchComparison();
                if (!replayMode) selectStep(steps.length - 1);
            } else if (msg.type === 'done') {
                generationActive = false;
                setStatus('done', `done · ${msg.total_steps} tokens`);
                generateBtn.disabled = false;
                if (!replayMode) selectHighestEntropyStep();
                updateBranchComparison();
            }
        };
        socket.onerror = () => {
            if (ws !== socket) return;
            generationActive = false;
            setStatus('error', 'connection error');
            generateBtn.disabled = false;
        };
    }

    generateBtn.addEventListener('click', () => startGeneration(promptInput.value));

    backToOriginalBtn.addEventListener('click', () => {
        if (!originalPrompt) return;
        maxTokensInput.value = originalMaxTokens;
        startGeneration(originalPrompt, { resetOriginal: false });
    });

    ProbBars.onPick((pickedToken) => {
        if (selectedStepIdx < 0 || !steps[selectedStepIdx]) return;
        const step = steps[selectedStepIdx];
        const comparison = { before: hauntPercent(steps) };
        const branchPrompt = runPrompt + steps
            .slice(0, selectedStepIdx)
            .map(previousStep => previousStep.token_str || '')
            .join('') + pickedToken;
        branchLabel.textContent = `branched at step ${selectedStepIdx + 1}: ${vis(step.token_str || '')} → ${vis(pickedToken)}`;
        maxTokensInput.value = '8';
        startGeneration(branchPrompt, { branch: true, resetOriginal: false, comparison });
    });

    function exorciseStep(index) {
        const step = steps[index];
        if (!step || !isGhost(step)) return;
        stepSlider.value = index;
        selectStep(index);
        const alternative = getAlternatives(step, 1)[0];
        if (!alternative || !alternative.token) return;

        const before = hauntPercent(steps);
        const branchPrompt = runPrompt + steps
            .slice(0, index)
            .map(previousStep => previousStep.token_str || '')
            .join('') + alternative.token;
        branchLabel.textContent = `exorcising step ${index + 1}: ${vis(step.token_str || '')} → ${vis(alternative.token)} · haunt before ${before}%`;
        maxTokensInput.value = '8';
        startGeneration(branchPrompt, {
            branch: true,
            resetOriginal: false,
            comparison: { before },
        });
    }

    function stopReplay() {
        if (replayTimer !== null) {
            window.clearInterval(replayTimer);
            replayTimer = null;
        }
        replayMode = false;
        replayPlay.textContent = 'play séance';
        replayAlternatives.hidden = true;
    }

    function updateReplayAlternatives() {
        replayAlternatives.replaceChildren();
        if (!replayMode || selectedStepIdx < 0 || !steps[selectedStepIdx]) {
            replayAlternatives.hidden = true;
            return;
        }
        const step = steps[selectedStepIdx];
        const alternatives = getAlternatives(step, 3);
        const heading = document.createElement('span');
        heading.className = 'replay-label';
        heading.textContent = 'ghost answers · other top candidates';
        replayAlternatives.appendChild(heading);
        alternatives.forEach(item => {
            const answer = document.createElement('span');
            answer.className = 'replay-ghost-answer';
            answer.textContent = `${item.token.replace(/\s+/g, ' ') || '∅'} · ${(item.probability * 100).toFixed(1)}%`;
            replayAlternatives.appendChild(answer);
        });
        replayAlternatives.hidden = false;
    }

    function toggleReplay() {
        if (!steps.length) return;
        if (replayTimer !== null) {
            window.clearInterval(replayTimer);
            replayTimer = null;
            replayPlay.textContent = 'resume séance';
            return;
        }
        const startingReplay = !replayMode;
        replayMode = true;
        replayPlay.textContent = 'pause séance';
        if (startingReplay || selectedStepIdx < 0) selectStep(0);
        replayTimer = window.setInterval(() => {
            if (selectedStepIdx + 1 < steps.length) {
                stepSlider.value = selectedStepIdx + 1;
                selectStep(selectedStepIdx + 1);
            } else if (!generationActive) {
                stopReplay();
            }
        }, 800);
        updateReplayAlternatives();
    }

    function stepReplay() {
        if (!steps.length) return;
        replayMode = true;
        if (replayTimer !== null) {
            window.clearInterval(replayTimer);
            replayTimer = null;
        }
        replayPlay.textContent = 'play séance';
        const nextStep = selectedStepIdx < 0 ? 0 : Math.min(selectedStepIdx + 1, steps.length - 1);
        stepSlider.value = nextStep;
        selectStep(nextStep);
    }

    ghostThreshold.addEventListener('input', () => {
        const thresholds = ghostThresholds();
        ghostThresholdValue.textContent = `${Number(ghostThreshold.value)}%`;
        ghostRule.textContent = `Ghost rule: H ≥ ${thresholds.entropy.toFixed(1)} bits AND raw P < ${(thresholds.probability * 100).toFixed(1)}%.`;
        renderTokenStream(selectedStepIdx);
        if (selectedStepIdx >= 0) selectStep(selectedStepIdx);
    });

    confidenceTrail.addEventListener('click', event => {
        if (!steps.length) return;
        const bounds = confidenceTrail.getBoundingClientRect();
        const left = 38, right = 142;
        const plotWidth = Math.max(1, bounds.width - left - right);
        const x = event.clientX - bounds.left - left;
        const index = steps.length <= 1
            ? 0
            : Math.max(0, Math.min(steps.length - 1, Math.round(x * (steps.length - 1) / plotWidth)));
        stepSlider.value = index;
        selectStep(index);
    });

    replayPlay.addEventListener('click', toggleReplay);
    replayStep.addEventListener('click', stepReplay);
    replayStop.addEventListener('click', stopReplay);
    spookyMode.addEventListener('change', () => {
        document.body.classList.toggle('spooky-mode', spookyMode.checked);
    });
    document.body.classList.toggle('spooky-mode', spookyMode.checked);
    window.addEventListener('resize', renderConfidenceTrail);
    window.addEventListener('resize', updateAttention);

    document.getElementById('head-gallery-open').addEventListener('click', () => {
        headGallery.showModal();
        renderHeadGallery();
    });
    document.getElementById('head-gallery-close').addEventListener('click', () => headGallery.close());
    headGallery.addEventListener('click', (event) => {
        if (event.target === headGallery) headGallery.close();
    });

    attnLayerSel.addEventListener('change', updateAttention);
    attnHeadSel.addEventListener('change', updateAttention);
    attnAvgHeads.addEventListener('change', updateAttention);
    attnRowNorm.addEventListener('change', updateAttention);
    attnHideSink.addEventListener('change', updateAttention);
    underhood.addEventListener('toggle', () => {
        if (underhood.open) window.requestAnimationFrame(updateAttention);
    });

    function applyDemoExample(id) {
        const example = demoExamples && demoExamples[id];
        if (!example) return;
        stopReplay();
        if (ws) ws.close();
        ws = null;
        generationActive = false;
        generateBtn.disabled = false;
        promptInput.value = example.prompt;
        runPrompt = example.prompt;
        promptTokens = example.prompt_token_strs || [];
        steps = example.steps || [];
        genTokens = steps.map(step => step.token_str || '');
        allAttentionData = [];
        selectedStepIdx = -1;
        branchBanner.hidden = true;
        branchComparison = null;
        stepSlider.max = Math.max(0, steps.length - 1);
        metricSpeed.textContent = steps.length
            ? (steps.reduce((sum, step) => sum + Number(step.tokens_per_sec || 0), 0) / steps.length).toFixed(1)
            : '—';
        metricCount.textContent = steps.filter(step => GhostSignals.isCountableToken(step.token_str)).length;
        metricCache.textContent = 'saved';
        traceSourceLabel.textContent = `${example.label} · saved trace · generate to run live.`;
        setStatus('idle', 'saved example');
        renderTokenStream();
        if (steps.length) selectHighestEntropyStep();
        else clearSelectedWord();
    }

    async function loadDemoTrace() {
        const response = await fetch('/static/demo-trace.json?v=2');
        if (!response.ok) throw new Error(`Could not load saved example (${response.status}).`);
        const trace = await response.json();
        if (trace.version !== 1 || !trace.examples || !trace.examples[trace.default]) {
            throw new Error('The saved example trace has an unsupported format.');
        }
        demoExamples = trace.examples;
        applyDemoExample(trace.default);
    }

    document.querySelectorAll('[data-demo-example]').forEach(button => {
        button.addEventListener('click', () => applyDemoExample(button.dataset.demoExample));
    });

    stepSlider.addEventListener('input', (e) => selectStep(parseInt(e.target.value)));

    tokenStream.addEventListener('click', (e) => {
        const exorciseButton = e.target.closest('[data-exorcise-step]');
        if (exorciseButton) {
            exorciseStep(Number(exorciseButton.dataset.exorciseStep));
            return;
        }
        const chip = e.target.closest('[data-step]');
        const stepAttr = chip && chip.getAttribute('data-step');
        if (stepAttr !== null) {
            const idx = parseInt(stepAttr);
            stepSlider.value = idx;
            selectStep(idx);
        }
    });

    setStatus('idle', 'idle');
    loadDemoTrace().catch(error => {
        console.error('Unable to load the saved example trace:', error);
        traceSourceLabel.textContent = 'Saved example unavailable · generate to run a prompt.';
        setStatus('error', 'example unavailable');
    });
})();
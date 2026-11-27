/* GlassBox Attention Heatmap — Full Width Canvas + Row Norm without Sink Bias */
/* Made with 🔮 by Vivi */

const Heatmap = (() => {
    const canvas = document.getElementById('attn-canvas');
    const ctx = canvas.getContext('2d');
    const info = document.getElementById('attn-info');

    let stepsData = [];
    let promptTokens = [];
    let genTokens = [];
    let lastRender = null;
    let resizeFrame = null;

    function viridis(t) {
        const stops = [[41, 18, 63], [103, 58, 183], [176, 86, 200], [242, 123, 66], [255, 200, 87]];
        t = Math.max(0, Math.min(1, t));
        const idx = t * (stops.length - 1);
        const i0 = Math.floor(idx), i1 = Math.min(i0 + 1, stops.length - 1);
        const fraction = idx - i0;
        const start = stops[i0], end = stops[i1];
        return `rgb(${Math.round(start[0] + (end[0] - start[0]) * fraction)},${Math.round(start[1] + (end[1] - start[1]) * fraction)},${Math.round(start[2] + (end[2] - start[2]) * fraction)})`;
    }

    function drawMini(miniCanvas, sourceMatrix, rowNorm = true, hideSink = true) {
        const miniCtx = miniCanvas.getContext('2d');
        const width = miniCanvas.width;
        const height = miniCanvas.height;
        const totalCols = Math.max(0, ...sourceMatrix.map(row => row.length));
        miniCtx.fillStyle = '#090a0f';
        miniCtx.fillRect(0, 0, width, height);
        if (!sourceMatrix.length || !totalCols) return;

        const cellW = width / totalCols;
        const cellH = height / sourceMatrix.length;
        sourceMatrix.forEach((sourceRow, rowIdx) => {
            const row = [...sourceRow];
            if (hideSink && row.length) row[0] = 0;
            const scaleValues = hideSink && row.length > 1 ? row.slice(1) : row;
            const maxValue = Math.max(0, ...scaleValues);
            row.forEach((value, colIdx) => {
                const normalized = rowNorm && !(hideSink && colIdx === 0) && maxValue > 0
                    ? value / maxValue
                    : value;
                miniCtx.fillStyle = hideSink && colIdx === 0 ? '#292d35' : viridis(Math.sqrt(normalized));
                miniCtx.fillRect(colIdx * cellW, rowIdx * cellH, cellW + 0.5, cellH + 0.5);
            });
        });
    }

    function render(data, pTokens, gTokens, layerIdx, headIdx, avgHeads, rowNorm, hideSink) {
        stepsData = data; promptTokens = pTokens; genTokens = gTokens;
        lastRender = [data, pTokens, gTokens, layerIdx, headIdx, avgHeads, rowNorm, hideSink];
        if (!stepsData || stepsData.length === 0) return;

        // Build raw matrix
        let matrix = [];
        for (const stepAttn of stepsData) {
            if (!stepAttn || !stepAttn[layerIdx]) continue;
            const layer = stepAttn[layerIdx];
            let row;
            if (avgHeads) {
                const nHeads = layer.length, len = layer[0].length;
                row = new Array(len).fill(0);
                for (let h = 0; h < nHeads; h++) {
                    for (let i = 0; i < len; i++) row[i] += layer[h][i] / nHeads;
                }
            } else {
                row = layer[headIdx] || [];
            }
            matrix.push([...row]);
        }

        if (matrix.length === 0) return;

        // 1. Hide Sink Option (Zero out Column 0)
        if (hideSink) {
            matrix = matrix.map(row => {
                const r = [...row];
                if (r.length > 0) r[0] = 0;
                return r;
            });
        }

        // 2. Row Normalization (EXCLUDING Column 0 if sink is hidden/ignored)
        if (rowNorm) {
            matrix = matrix.map(row => {
                // If hideSink is ON, exclude col 0 from max calculation so non-sink weights scale 0..1!
                const searchSlice = (hideSink && row.length > 1) ? row.slice(1) : row;
                const maxVal = Math.max(...searchSlice);
                if (maxVal > 0) {
                    return row.map((v, colIdx) => (colIdx === 0 && hideSink) ? 0 : v / maxVal);
                }
                return row;
            });
        }

        // Apply Sqrt scaling so subtle attention signals pop out vividly
        const scaledMatrix = matrix.map(row => row.map(v => Math.sqrt(v)));

        // Setup Canvas Size (Full Width Stretch)
        const totalRows = matrix.length;
        const totalCols = matrix[matrix.length - 1].length;

        const wrapper = canvas.parentElement;
        const styles = getComputedStyle(wrapper);
        const horizontalSpace = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
        const verticalSpace = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom)
            + parseFloat(styles.borderTopWidth) + parseFloat(styles.borderBottomWidth);
        const parentW = Math.max(1, wrapper.clientWidth - horizontalSpace);
        const marginLeft = 112;
        const marginTop = 92;
        const drawW = Math.max(1, parentW - marginLeft - 10);
        const cellSize = Math.max(1, Math.min(46, drawW / totalCols));
        const parentH = marginTop + totalRows * cellSize + 8;
        wrapper.style.height = `${parentH + verticalSpace}px`;
        canvas.width = parentW;
        canvas.height = parentH;

        ctx.fillStyle = '#090a0f';
        ctx.fillRect(0, 0, parentW, parentH);

        // Render Heatmap Cells
        for (let i = 0; i < totalRows; i++) {
            for (let j = 0; j < scaledMatrix[i].length; j++) {
                ctx.fillStyle = viridis(scaledMatrix[i][j]);
                ctx.fillRect(marginLeft + j * cellSize, marginTop + i * cellSize, cellSize - 1, cellSize - 1);
            }
        }

        ctx.strokeStyle = '#ff7a1a';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            marginLeft + 1,
            marginTop + (totalRows - 1) * cellSize + 1,
            Math.max(0, totalCols * cellSize - 2),
            Math.max(0, cellSize - 2)
        );

        if (hideSink && totalCols > 0) {
            const sinkX = marginLeft;
            const sinkWidth = cellSize;
            ctx.save();
            ctx.beginPath();
            ctx.rect(sinkX, marginTop, sinkWidth, totalRows * cellSize);
            ctx.clip();
            ctx.fillStyle = '#292d35';
            ctx.fillRect(sinkX, marginTop, sinkWidth, totalRows * cellSize);
            ctx.strokeStyle = '#737b88';
            ctx.lineWidth = 1;
            for (let offset = -totalRows * cellSize; offset < sinkWidth; offset += 8) {
                ctx.beginPath();
                ctx.moveTo(sinkX + offset, marginTop + totalRows * cellSize);
                ctx.lineTo(sinkX + offset + totalRows * cellSize, marginTop);
                ctx.stroke();
            }
            ctx.restore();
        }

        // Keep every column labeled; shorten labels only as cells get narrower.
        ctx.fillStyle = '#b9a8d6';
        const labelFontSize = Math.min(12, Math.max(8, cellSize * 0.26));
        ctx.font = `${labelFontSize}px JetBrains Mono`;
        const allTokens = [...promptTokens, ...genTokens];
        const maxLabelLength = Math.max(1, Math.min(8, Math.floor(cellSize * Math.SQRT2 / (labelFontSize * 0.6))));
        for (let j = 0; j < totalCols; j++) {
            const label = readableToken(allTokens[j] || '').slice(0, maxLabelLength);
            ctx.save();
            ctx.translate(marginLeft + (j + 0.5) * cellSize, marginTop - 8);
            ctx.rotate(-Math.PI / 4);
            ctx.textAlign = 'left';
            ctx.fillText(label || '∅', 0, 0, 64);
            ctx.restore();
        }

        // Each attention row predicts the generated token at the same step index.
        const rowStep = Math.max(1, Math.ceil(18 / cellSize));
        ctx.textAlign = 'right';
        for (let i = 0; i < totalRows; i += rowStep) {
            const predictedToken = genTokens[i] || '';
            const rowLabel = readableToken(predictedToken).slice(0, 12);
            ctx.fillText(rowLabel || '∅', marginLeft - 8, marginTop + (i + 0.72) * cellSize, marginLeft - 12);
        }

        // Hover Tooltip Logic
        canvas.onmousemove = (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left - marginLeft;
            const y = e.clientY - rect.top - marginTop;
            const col = Math.floor(x / cellSize);
            const row = Math.floor(y / cellSize);

            if (row >= 0 && row < matrix.length && col >= 0 && col < matrix[row].length) {
                const layerRows = stepsData[row] && stepsData[row][layerIdx] ? stepsData[row][layerIdx] : [];
                const selectedRow = avgHeads
                    ? (layerRows[0] || []).map((_, index) => layerRows.reduce((sum, values) => sum + values[index], 0) / layerRows.length)
                    : (layerRows[headIdx] || []);
                const rawVal = selectedRow[col] || 0;
                const rowMaxValues = hideSink ? selectedRow.slice(1) : selectedRow;
                const rowMax = rowMaxValues.length ? Math.max(...rowMaxValues) : 0;
                const percentOfMax = rowMax > 0 ? (rawVal / rowMax) * 100 : 0;
                const predictedTok = genTokens[row] || '';
                const toTok = allTokens[col] || `col ${col}`;
                const cleanPrediction = readableToken(predictedTok) || '∅';
                const cleanKey = readableToken(toTok) || '∅';
                info.textContent = `Row ${row + 1}/${totalRows}: wrote '${cleanPrediction}' and attended to earlier token ${col + 1}, '${cleanKey}'; raw ${rawVal.toFixed(4)}, ${percentOfMax.toFixed(1)}% of row max`;
            }
        };
    }

    function readableToken(token) {
        if (token === '\n' || token === '\r\n' || token === '\r') return 'newline';
        if (token === ' ') return '␠';
        return String(token).replace(/[\r\n\t]/g, '↵').trim();
    }

    window.addEventListener('resize', () => {
        if (!lastRender || resizeFrame !== null) return;
        resizeFrame = window.requestAnimationFrame(() => {
            resizeFrame = null;
            render(...lastRender);
        });
    });

    return { render, drawMini };
})();
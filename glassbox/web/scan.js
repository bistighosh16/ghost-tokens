/* Haunt Scan UI helpers. */
window.HauntScan = (() => {
    const presets = {
        ordinary: 'The sun rises in the east and sets in the west.',
        false: 'The capital of France is Berlin.',
        gibberish: 'Zqorvax Blenthwick calibrated the moonlit hypersonic teapot.',
    };

    function thresholdForSensitivity(sensitivity) {
        const level = Math.min(100, Math.max(0, Number(sensitivity))) / 100;
        return 14 - 9 * level;
    }

    function isGhost(token, sensitivity) {
        return window.GhostSignals.isCountableToken(token.token_str)
            && Number(token.surprise_bits) >= thresholdForSensitivity(sensitivity);
    }

    function hauntPercent(tokens, sensitivity) {
        const countable = tokens.filter(token => window.GhostSignals.isCountableToken(token.token_str));
        if (!countable.length) return 0;
        return Math.round(100 * countable.filter(token => isGhost(token, sensitivity)).length / countable.length);
    }

    return { presets, thresholdForSensitivity, isGhost, hauntPercent };
})();

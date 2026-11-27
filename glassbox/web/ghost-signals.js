/* Shared browser rules for live traces and the static demo. */
window.GhostSignals = (() => {
    function thresholdsForSensitivity(sensitivity = 50) {
        const level = Math.min(100, Math.max(0, Number(sensitivity))) / 100;
        return {
            entropy: 6 - 3 * level,
            probability: 0.02 + 0.26 * level,
        };
    }

    function isCountableToken(token) {
        return /[\p{L}\p{N}]/u.test(token || '');
    }

    function isGhost(step, sensitivity = 50) {
        if (!isCountableToken(step.token_str)) return false;
        const thresholds = thresholdsForSensitivity(sensitivity);
        return Number(step.entropy_bits) >= thresholds.entropy
            && Number(step.p_chosen) < thresholds.probability;
    }

    function hauntPercent(items, sensitivity = 50) {
        const countable = items.filter(step => isCountableToken(step.token_str));
        if (!countable.length) return 0;
        const numerator = 100 * countable.filter(step => isGhost(step, sensitivity)).length;
        const whole = Math.floor(numerator / countable.length);
        const remainder = numerator % countable.length;
        if (remainder * 2 < countable.length) return whole;
        if (remainder * 2 > countable.length) return whole + 1;
        return whole % 2 === 0 ? whole : whole + 1;
    }

    return { thresholdsForSensitivity, isCountableToken, isGhost, hauntPercent };
})();

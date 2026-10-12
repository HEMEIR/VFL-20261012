def buildRobustnessMetrics(asr, attackedSamples, triggeredBa=None):
    return {
        "asr": asr,
        "attackedSamples": attackedSamples,
        "triggeredBa": triggeredBa,
    }

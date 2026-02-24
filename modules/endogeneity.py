def endogeneity_score(hausman_p, max_vif, bp_p):

    score = 0

    if hausman_p is not None and hausman_p < 0.05:
        score += 30

    if max_vif > 10:
        score += 20

    if bp_p < 0.05:
        score += 10

    if score < 30:
        level = "Low"
    elif score < 60:
        level = "Moderate"
    else:
        level = "High"

    return score, level


def suggest_instruments(variable):

    suggestions = {
        "fdi": ["Lagged FDI", "Global FDI shocks"],
        "gdppc": ["Lagged GDP per capita"],
        "institutions": ["Legal origin", "Colonial history"]
    }

    return suggestions.get(variable.lower(), ["Lagged values"])

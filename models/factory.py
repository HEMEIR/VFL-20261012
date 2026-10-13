from .image_bottom import ImageBottomModel
from .structured_bottom import StructuredBottomModel
from .top_model import TopModel


def buildModels(bundle, modelConfig):
    embeddingDim = int(modelConfig["embeddingDim"])
    bottomType = str(modelConfig["bottomType"])
    if bottomType == "resnet18":
        bottomA = ImageBottomModel(embeddingDim)
        bottomB = ImageBottomModel(embeddingDim)
    elif bottomType == "mlp":
        hiddenDims = [int(dimension) for dimension in modelConfig["bottomHiddenDims"]]
        bottomA = StructuredBottomModel(bundle.inputDimA, hiddenDims, embeddingDim)
        bottomB = StructuredBottomModel(bundle.inputDimB, hiddenDims, embeddingDim)
    else:
        raise ValueError(f"Unsupported bottom model type: {bottomType}")
    # The top model's output width can exceed the class count: the alias defence gives each
    # true class a BLOCK of K alias labels, so the model must be able to emit
    # classCount * K indices.  Defaults to 1, i.e. behaviour is unchanged unless set.
    classCount = getattr(bundle, "classCount", 2)
    aliasFactor = max(1, int(getattr(bundle, "labelAliasFactor", 1) or 1))
    # With aliasing the output width must hold BOTH the true label range and the alias
    # blocks that sit above it: offset = classCount * K, aliases occupy another classCount * K,
    # so the width is 2 * classCount * K.  Using classCount * K made the aliases exceed the
    # output width and CUDA aborted with "device-side assert triggered".
    # Output width = offset + classCount * aliasFactor, with the offset equal to
    # classCount * aliasFactor so aliases sit entirely ABOVE the true label range (the
    # requirement not to use 0/1).  This must match the defence's own offset derivation.
    # K == 1 IS THE BASELINE: no aliasing, no offset, width must equal classCount.  Applying
    # the offset unconditionally changed the top model's width from 2 to 4 for EVERY
    # configuration including the baseline, which silently invalidated a whole pilot - the
    # baseline no longer reproduced its own earlier numbers.
    if bool(getattr(bundle, "labelAliasSharePeaks", False)):
        # Shared peaks: C private positions + C//2 shared positions (partial), or just C
        # positions when full overlap (labelAliasShareAllPeaks).
        if bool(getattr(bundle, "labelAliasShareAllPeaks", False)):
            width = classCount
        else:
            width = classCount + classCount // 2
    elif aliasFactor <= 1:
        width = classCount
    elif bool(getattr(bundle, "labelAliasSymmetric", False)):
        # Symmetric (edge-to-centre) aliases use NO offset: the peaks span [0, C*K).
        doubleLevel = max(1, int(getattr(bundle, "labelAliasDoubleLevel", 1) or 1))
        width = classCount * aliasFactor * doubleLevel
    else:
        width = classCount * aliasFactor * 2
    return bottomA, bottomB, TopModel(
        embeddingDim,
        width,
    )

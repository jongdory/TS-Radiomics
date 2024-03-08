import inspect
import logging
import traceback
import numpy
import torch
import SimpleITK as sitk
from radiomics import base, cShape, deprecated, getProgressReporter, imageoperations
import six


class RadiomicsFeaturesBase(object):
  """
  This is the abstract class, which defines the common interface for the feature classes. All feature classes inherit
  (directly of indirectly) from this class.

  At initialization, image and labelmap are passed as SimpleITK image objects (``inputImage`` and ``inputMask``,
  respectively.) The motivation for using SimpleITK images as input is to keep the possibility of reusing the
  optimized feature calculators implemented in SimpleITK in the future. If either the image or the mask is None,
  initialization fails and a warning is logged (does not raise an error).

  Logging is set up using a child logger from the parent 'radiomics' logger. This retains the toolbox structure in
  the generated log. The child logger is named after the module containing the feature class (e.g. 'radiomics.glcm').

  Any pre calculations needed before the feature functions are called can be added by overriding the
  ``_initSegmentBasedCalculation`` function, which prepares the input for feature extraction. If image discretization is
  needed, this can be implemented by adding a call to ``_applyBinning`` to this initialization function, which also
  instantiates coefficients holding the maximum ('Ng') and unique ('GrayLevels') that can be found inside the ROI after
  binning. This function also instantiates the `matrix` variable, which holds the discretized image (the `imageArray`
  variable will hold only original gray levels).

  The following variables are instantiated at initialization:

  - kwargs: dictionary holding all customized settings passed to this feature class.
  - label: label value of Region of Interest (ROI) in labelmap. If key is not present, a default value of 1 is used.
  - featureNames: list containing the names of features defined in the feature class. See :py:func:`getFeatureNames`
  - inputImage: SimpleITK image object of the input image (dimensions x, y, z)

  The following variables are instantiated by the ``_initSegmentBasedCalculation`` function:

  - inputMask: SimpleITK image object of the input labelmap (dimensions x, y, z)
  - imageArray: numpy array of the gray values in the input image (dimensions z, y, x)
  - maskArray: numpy boolean array with elements set to ``True`` where labelmap = label, ``False`` otherwise,
    (dimensions z, y, x).
  - labelledVoxelCoordinates: tuple of 3 numpy arrays containing the z, x and y coordinates of the voxels included in
    the ROI, respectively. Length of each array is equal to total number of voxels inside ROI.
  - matrix: copy of the imageArray variable, with gray values inside ROI discretized using the specified binWidth.
    This variable is only instantiated if a call to ``_applyBinning`` is added to an override of
    ``_initSegmentBasedCalculation`` in the feature class.

  .. note::
    Although some variables listed here have similar names to customization settings, they do *not* represent all the
    possible settings on the feature class level. These variables are listed here to help developers develop new feature
    classes, which make use of these variables. For more information on customization, see
    :ref:`radiomics-customization-label`, which includes a comprehensive list of all possible settings, including
    default values and explanation of usage.
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    self.logger = logging.getLogger(self.__module__)
    self.logger.debug('Initializing feature class')

    if inputImage is None or inputMask is None:
      raise ValueError('Missing input image or mask')

    self.progressReporter = getProgressReporter

    self.settings = kwargs

    self.label = kwargs.get('label', 1)
    self.voxelBased = kwargs.get('voxelBased', False)

    self.coefficients = {}

    # all features are disabled by default
    self.enabledFeatures = {}
    self.featureValues = {}

    self.featureNames = self.getFeatureNames()

    self.inputImage = inputImage
    self.inputMask = inputMask

    self.imageArray = self.inputImage #sitk.GetArrayFromImage()

    if self.voxelBased:
      self._initVoxelBasedCalculation()
    else:
      self._initSegmentBasedCalculation()

  def _initSegmentBasedCalculation(self):
    self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)  # boolean array

  def _initVoxelBasedCalculation(self):
    self.masked = self.settings.get('maskedKernel', True)

    maskArray = sitk.GetArrayFromImage(self.inputMask) == self.label  # boolean array
    self.labelledVoxelCoordinates = numpy.array(numpy.where(maskArray))

    # Set up the mask array for the gray value discretization
    if self.masked:
      self.maskArray = maskArray
    else:
      # This will cause the discretization to use the entire image
      self.maskArray = numpy.ones(self.imageArray.shape, dtype='bool')

  def _initCalculation(self, voxelCoordinates=None):
    """
    Last steps to prepare the class for extraction. This function calculates the texture matrices and coefficients in
    the respective feature classes
    """
    pass

  def _applyBinning(self, matrix):
    matrix, _ = imageoperations.binImage(matrix, self.maskArray, **self.settings)
    self.coefficients['grayLevels'] = numpy.unique(matrix[self.maskArray])
    self.coefficients['Ng'] = int(numpy.max(self.coefficients['grayLevels']))  # max gray level in the ROI
    return matrix

  def enableFeatureByName(self, featureName, enable=True):
    """
    Enables or disables feature specified by ``featureName``. If feature is not present in this class, a lookup error is
    raised. ``enable`` specifies whether to enable or disable the feature.
    """
    if featureName not in self.featureNames:
      raise LookupError('Feature not found: ' + featureName)
    if self.featureNames[featureName]:
      self.logger.warning('Feature %s is deprecated, use with caution!', featureName)
    self.enabledFeatures[featureName] = enable

  def enableAllFeatures(self):
    """
    Enables all features found in this class for calculation.

    .. note::
      Features that have been marked "deprecated" are not enabled by this function. They can still be enabled manually by
      a call to :py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`,
      :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()`
      or in the parameter file (by specifying the feature by name, not when enabling all features).
      However, in most cases this will still result only in a deprecation warning.
    """
    for featureName, is_deprecated in six.iteritems(self.featureNames):
      # only enable non-deprecated features here
      if not is_deprecated:
        self.enableFeatureByName(featureName, True)

  def disableAllFeatures(self):
    """
    Disables all features. Additionally resets any calculated features.
    """
    self.enabledFeatures = {}
    self.featureValues = {}

  @classmethod
  def getFeatureNames(cls):
    """
    Dynamically enumerates features defined in the feature class. Features are identified by the
    ``get<Feature>FeatureValue`` signature, where <Feature> is the name of the feature (unique on the class level).

    Found features are returned as a dictionary of the feature names, where the value ``True`` if the
    feature is deprecated, ``False`` otherwise (``{<Feature1>:<deprecated>, <Feature2>:<deprecated>, ...}``).

    This function is called at initialization, found features are stored in the ``featureNames`` variable.
    """
    attributes = inspect.getmembers(cls)
    features = {a[0][3:-12]: getattr(a[1], '_is_deprecated', False) for a in attributes
                if a[0].startswith('get') and a[0].endswith('FeatureValue')}
    return features

  def execute(self):
    """
    Calculates all features enabled in  ``enabledFeatures``. A feature is enabled if it's key is present in this
    dictionary and it's value is True.

    Calculated values are stored in the ``featureValues`` dictionary, with feature name as key and the calculated
    feature value as value. If an exception is thrown during calculation, the error is logged, and the value is set to
    NaN.
    """
    if len(self.enabledFeatures) == 0:
      self.enableAllFeatures()

    if self.voxelBased:
      self._calculateVoxels()
    else:
      self._calculateSegment()

    return self.featureValues

  def _calculateVoxels(self):
    initValue = self.settings.get('initValue', 0)
    voxelBatch = self.settings.get('voxelBatch', -1)

    # Initialize the output with empty numpy arrays
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = numpy.full(list(self.inputImage.GetSize())[::-1], initValue, dtype='float')

    # Calculate the feature values for all enabled features
    voxel_count = self.labelledVoxelCoordinates.shape[1]
    voxel_batch_idx = 0
    if voxelBatch < 0:
      voxelBatch = voxel_count
    n_batches = numpy.ceil(float(voxel_count) / voxelBatch)
    with self.progressReporter(total=n_batches, desc='batch') as pbar:
      while voxel_batch_idx < voxel_count:
        self.logger.debug('Calculating voxel batch no. %i/%i', int(voxel_batch_idx / voxelBatch) + 1, n_batches)
        voxelCoords = self.labelledVoxelCoordinates[:, voxel_batch_idx:voxel_batch_idx + voxelBatch]
        # Calculate the feature values for the current kernel
        for success, featureName, featureValue in self._calculateFeatures(voxelCoords):
          if success:
            self.featureValues[featureName][tuple(voxelCoords)] = featureValue

        voxel_batch_idx += voxelBatch
        pbar.update(1)  # Update progress bar

    # Convert the output to simple ITK image objects
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = sitk.GetImageFromArray(self.featureValues[feature])
        self.featureValues[feature].CopyInformation(self.inputImage)

  def _calculateSegment(self):
    # Get the feature values using the current segment.
    for success, featureName, featureValue in self._calculateFeatures():
      # Always store the result. In case of an error, featureValue will be NaN
      self.featureValues[featureName] = numpy.squeeze(featureValue)

  def _calculateFeatures(self, voxelCoordinates=None):
    # Initialize the calculation
    # This function serves to calculate the texture matrices where applicable
    self._initCalculation(voxelCoordinates)

    self.logger.debug('Calculating features')
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        try:
          # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
          yield True, feature, getattr(self, 'get%sFeatureValue' % feature)()
        except DeprecationWarning as deprecatedFeature:
          # Add a debug log message, as a warning is usually shown and would entail a too verbose output
          self.logger.debug('Feature %s is deprecated: %s', feature, deprecatedFeature.args[0])
        except Exception:
          self.logger.error('FAILED: %s', traceback.format_exc())
          yield False, feature, numpy.nan

class RadiomicsShape(RadiomicsFeaturesBase):
  r"""
  In this group of features we included descriptors of the three-dimensional size and shape of the ROI. These features
  are independent from the gray level intensity distribution in the ROI and are therefore only calculated on the
  non-derived image and mask.

  Unless otherwise specified, features are derived from the approximated shape defined by the triangle mesh. To build
  this mesh, vertices (points) are first defined as points halfway on an edge between a voxel included in the ROI and
  one outside the ROI. By connecting these vertices a mesh of connected triangles is obtained, with each triangle
  defined by 3 adjacent vertices, which shares each side with exactly one other triangle.

  This mesh is generated using a marching cubes algorithm. In this algorithm, a 2x2 cube is moved through the mask
  space. For each position, the corners of the cube are then marked 'segmented' (1) or 'not segmented' (0). Treating the
  corners as specific bits in a binary number, a unique cube-index is obtained (0-255). This index is then used to
  determine which triangles are present in the cube, which are defined in a lookup table.

  These triangles are defined in such a way, that the normal (obtained from the cross product of vectors describing 2
  out of 3 edges) are always oriented in the same direction. For PyRadiomics, the calculated normals are always pointing
  outward. This is necessary to obtain the correct signed volume used in calculation of ``MeshVolume``.

  Let:

  - :math:`N_v` represent the number of voxels included in the ROI
  - :math:`N_f` represent the number of faces (triangles) defining the Mesh.
  - :math:`V` the volume of the mesh in mm\ :sup:`3`, calculated by :py:func:`getMeshVolumeFeatureValue`
  - :math:`A` the surface area of the mesh in mm\ :sup:`2`, calculated by :py:func:`getMeshSurfaceAreaFeatureValue`

  References:

  - Lorensen WE, Cline HE. Marching cubes: A high resolution 3D surface construction algorithm. ACM SIGGRAPH Comput
    Graph `Internet <http://portal.acm.org/citation.cfm?doid=37402.37422>`_. 1987;21:163-9.
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    # assert inputMask.GetDimension() == 3, 'Shape features are only available in 3D. If 2D, use shape2D instead'
    super(RadiomicsShape, self).__init__(inputImage, inputMask, **kwargs)

  def _initVoxelBasedCalculation(self):
    raise NotImplementedError('Shape features are not available in voxel-based mode')

  def _initSegmentBasedCalculation(self):

    self.pixelSpacing = numpy.array((1,1,1)) #torch.array(self.inputImage.GetSpacing()[::-1])

    # Pad inputMask to prevent index-out-of-range errors
    self.logger.debug('Padding the mask with 0s')

    # cpif = sitk.ConstantPadImageFilter()

    # padding = torch.tile(1, 3)
    # try:
    #   cpif.SetPadLowerBound(padding)
    #   cpif.SetPadUpperBound(padding)
    # except TypeError:
    #   # newer versions of SITK/python want a tuple or list
    #   cpif.SetPadLowerBound(padding.tolist())
    #   cpif.SetPadUpperBound(padding.tolist())

    # self.inputMask = cpif.Execute(self.inputMask)

    # Reassign self.maskArray using the now-padded self.inputMask
    self.maskArray = self.inputMask#(sitk.GetArrayFromImage(self.inputMask) == self.label)
    self.labelledVoxelCoordinates = torch.where(self.maskArray != 0)

    self.logger.debug('Pre-calculate Volume, Surface Area and Eigenvalues')

    # Volume, Surface Area and eigenvalues are pre-calculated

    # Compute Surface Area and volume
    self.SurfaceArea, self.Volume, self.diameters = cShape.calculate_coefficients(self.maskArray, self.pixelSpacing)

    # Compute eigenvalues and -vectors
    Np = len(self.labelledVoxelCoordinates[0])
    coordinates = torch.array(self.labelledVoxelCoordinates, dtype='int').transpose((1, 0))  # Transpose equals zip(*a)
    physicalCoordinates = coordinates * self.pixelSpacing[None, :]
    physicalCoordinates -= torch.mean(physicalCoordinates, axis=0)  # Centered at 0
    physicalCoordinates /= torch.sqrt(Np)
    covariance = torch.dot(physicalCoordinates.T.copy(), physicalCoordinates)
    self.eigenValues = torch.linalg.eigvals(covariance)

    # Correct machine precision errors causing very small negative eigen values in case of some 2D segmentations
    machine_errors = torch.bitwise_and(self.eigenValues < 0, self.eigenValues > -1e-10)
    if torch.sum(machine_errors) > 0:
      self.logger.warning('Encountered %d eigenvalues < 0 and > -1e-10, rounding to 0', torch.sum(machine_errors))
      self.eigenValues[machine_errors] = 0

    self.eigenValues.sort()  # Sort the eigenValues from small to large

    self.logger.debug('Shape feature class initialized')

def getMeshVolumeFeatureValue(self):
    r"""
    **1. Mesh Volume**

    .. math::
      V_i = \displaystyle\frac{Oa_i \cdot (Ob_i \times Oc_i)}{6} \text{ (1)}

      V = \displaystyle\sum^{N_f}_{i=1}{V_i} \text{ (2)}

    The volume of the ROI :math:`V` is calculated from the triangle mesh of the ROI.
    For each face :math:`i` in the mesh, defined by points :math:`a_i, b_i` and :math:`c_i`, the (signed) volume
    :math:`V_f` of the tetrahedron defined by that face and the origin of the image (:math:`O`) is calculated. (1)
    The sign of the volume is determined by the sign of the normal, which must be consistently defined as either facing
    outward or inward of the ROI.

    Then taking the sum of all :math:`V_i`, the total volume of the ROI is obtained (2)

    .. note::
      For more extensive documentation on how the volume is obtained using the surface mesh, see the IBSI document,
      where this feature is defined as ``Volume``.
    """
    return self.Volume


def getVoxelVolumeFeatureValue(self):
    r"""
    **2. Voxel Volume**

    .. math::
      V_{voxel} = \displaystyle\sum^{N_v}_{k=1}{V_k}

    The volume of the ROI :math:`V_{voxel}` is approximated by multiplying the number of voxels in the ROI by the volume
    of a single voxel :math:`V_k`. This is a less precise approximation of the volume and is not used in subsequent
    features. This feature does not make use of the mesh and is not used in calculation of other shape features.

    .. note::
      Defined in IBSI as ``Approximate Volume``.
    """
    z, y, x = self.pixelSpacing
    Np = len(self.labelledVoxelCoordinates[0])
    return Np * (z * x * y)


def getSurfaceAreaFeatureValue(self):
    r"""
    **3. Surface Area**

    .. math::
      A_i = \frac{1}{2}|\text{a}_i\text{b}_i \times \text{a}_i\text{c}_i| \text{ (1)}

      A = \displaystyle\sum^{N_f}_{i=1}{A_i} \text{ (2)}

    where:

    :math:`\text{a}_i\text{b}_i` and :math:`\text{a}_i\text{c}_i` are edges of the :math:`i^{\text{th}}` triangle in the
    mesh, formed by vertices :math:`\text{a}_i`, :math:`\text{b}_i` and :math:`\text{c}_i`.

    To calculate the surface area, first the surface area :math:`A_i` of each triangle in the mesh is calculated (1).
    The total surface area is then obtained by taking the sum of all calculated sub-areas (2).

    .. note::
      Defined in IBSI as ``Surface Area``.
    """
    return self.SurfaceArea


def getSurfaceVolumeRatioFeatureValue(self):
    r"""
    **4. Surface Area to Volume ratio**

    .. math::
      \textit{surface to volume ratio} = \frac{A}{V}

    Here, a lower value indicates a more compact (sphere-like) shape. This feature is not dimensionless, and is
    therefore (partly) dependent on the volume of the ROI.
    """
    return self.SurfaceArea / self.Volume


def getSphericityFeatureValue(self):
    r"""
    **5. Sphericity**

    .. math::
      \textit{sphericity} = \frac{\sqrt[3]{36 \pi V^2}}{A}

    Sphericity is a measure of the roundness of the shape of the tumor region relative to a sphere. It is a
    dimensionless measure, independent of scale and orientation. The value range is :math:`0 < sphericity \leq 1`, where
    a value of 1 indicates a perfect sphere (a sphere has the smallest possible surface area for a given volume,
    compared to other solids).

    .. note::
      This feature is correlated to Compactness 1, Compactness 2 and Spherical Disproportion. In the default
      parameter file provided in the ``pyradiomics/examples/exampleSettings`` folder, Compactness 1 and Compactness 2
      are therefore disabled.
    """
    return (36 * numpy.pi * self.Volume ** 2) ** (1.0 / 3.0) / self.SurfaceArea


@deprecated
def getCompactness1FeatureValue(self):
    r"""
    **6. Compactness 1**

    .. math::
      \textit{compactness 1} = \frac{V}{\sqrt{\pi A^3}}

    Similar to Sphericity, Compactness 1 is a measure of how compact the shape of the tumor is relative to a sphere
    (most compact). It is therefore correlated to Sphericity and redundant. It is provided here for completeness.
    The value range is :math:`0 < compactness\ 1 \leq \frac{1}{6 \pi}`, where a value of :math:`\frac{1}{6 \pi}`
    indicates a perfect sphere.

    By definition, :math:`compactness\ 1 = \frac{1}{6 \pi}\sqrt{compactness\ 2} =
    \frac{1}{6 \pi}\sqrt{sphericity^3}`.

    .. note::
      This feature is correlated to Compactness 2, Sphericity and Spherical Disproportion.
      Therefore, this feature is marked, so it is not enabled by default (i.e. this feature will not be enabled if no
      individual features are specified (enabling 'all' features), but will be enabled when individual features are
      specified, including this feature). To include this feature in the extraction, specify it by name in the enabled
      features.
    """
    return self.Volume / (self.SurfaceArea ** (3.0 / 2.0) * torch.sqrt(numpy.pi))


@deprecated
def getCompactness2FeatureValue(self):
    r"""
    **7. Compactness 2**

    .. math::
      \textit{compactness 2} = 36 \pi \frac{V^2}{A^3}

    Similar to Sphericity and Compactness 1, Compactness 2 is a measure of how compact the shape of the tumor is
    relative to a sphere (most compact). It is a dimensionless measure, independent of scale and orientation. The value
    range is :math:`0 < compactness\ 2 \leq 1`, where a value of 1 indicates a perfect sphere.

    By definition, :math:`compactness\ 2 = (sphericity)^3`

    .. note::
      This feature is correlated to Compactness 1, Sphericity and Spherical Disproportion.
      Therefore, this feature is marked, so it is not enabled by default (i.e. this feature will not be enabled if no
      individual features are specified (enabling 'all' features), but will be enabled when individual features are
      specified, including this feature). To include this feature in the extraction, specify it by name in the enabled
      features.
    """
    return (36.0 * numpy.pi) * (self.Volume ** 2.0) / (self.SurfaceArea ** 3.0)


@deprecated
def getSphericalDisproportionFeatureValue(self):
    r"""
    **8. Spherical Disproportion**

    .. math::
      \textit{spherical disproportion} = \frac{A}{4\pi R^2} = \frac{A}{\sqrt[3]{36 \pi V^2}}

    Where :math:`R` is the radius of a sphere with the same volume as the tumor, and equal to
    :math:`\sqrt[3]{\frac{3V}{4\pi}}`.

    Spherical Disproportion is the ratio of the surface area of the tumor region to the surface area of a sphere with
    the same volume as the tumor region, and by definition, the inverse of Sphericity. Therefore, the value range is
    :math:`spherical\ disproportion \geq 1`, with a value of 1 indicating a perfect sphere.

    .. note::
      This feature is correlated to Compactness 2, Compactness2 and Sphericity.
      Therefore, this feature is marked, so it is not enabled by default (i.e. this feature will not be enabled if no
      individual features are specified (enabling 'all' features), but will be enabled when individual features are
      specified, including this feature). To include this feature in the extraction, specify it by name in the enabled
      features.
    """
    return self.SurfaceArea / (36 * numpy.pi * self.Volume ** 2) ** (1.0 / 3.0)


def getMaximum3DDiameterFeatureValue(self):
    r"""
    **9. Maximum 3D diameter**

    Maximum 3D diameter is defined as the largest pairwise Euclidean distance between tumor surface mesh
    vertices.

    Also known as Feret Diameter.
    """
    return self.diameters[3]


def getMaximum2DDiameterSliceFeatureValue(self):
    r"""
    **10. Maximum 2D diameter (Slice)**

    Maximum 2D diameter (Slice) is defined as the largest pairwise Euclidean distance between tumor surface mesh
    vertices in the row-column (generally the axial) plane.
    """
    return self.diameters[0]


def getMaximum2DDiameterColumnFeatureValue(self):
    r"""
    **11. Maximum 2D diameter (Column)**

    Maximum 2D diameter (Column) is defined as the largest pairwise Euclidean distance between tumor surface mesh
    vertices in the row-slice (usually the coronal) plane.
    """
    return self.diameters[1]


def getMaximum2DDiameterRowFeatureValue(self):
    r"""
    **12. Maximum 2D diameter (Row)**

    Maximum 2D diameter (Row) is defined as the largest pairwise Euclidean distance between tumor surface mesh
    vertices in the column-slice (usually the sagittal) plane.
    """
    return self.diameters[2]


def getMajorAxisLengthFeatureValue(self):
    r"""
    **13. Major Axis Length**

    .. math::
      \textit{major axis} = 4 \sqrt{\lambda_{major}}

    This feature yield the largest axis length of the ROI-enclosing ellipsoid and is calculated using the largest
    principal component :math:`\lambda_{major}`.

    The principal component analysis is performed using the physical coordinates of the voxel centers defining the ROI.
    It therefore takes spacing into account, but does not make use of the shape mesh.
    """
    if self.eigenValues[2] < 0:
      self.logger.warning('Major axis eigenvalue negative! (%g)', self.eigenValues[2])
      return numpy.nan
    return torch.sqrt(self.eigenValues[2]) * 4


def getMinorAxisLengthFeatureValue(self):
    r"""
    **14. Minor Axis Length**

    .. math::
      \textit{minor axis} = 4 \sqrt{\lambda_{minor}}

    This feature yield the second-largest axis length of the ROI-enclosing ellipsoid and is calculated using the largest
    principal component :math:`\lambda_{minor}`.

    The principal component analysis is performed using the physical coordinates of the voxel centers defining the ROI.
    It therefore takes spacing into account, but does not make use of the shape mesh.
    """
    if self.eigenValues[1] < 0:
      self.logger.warning('Minor axis eigenvalue negative! (%g)', self.eigenValues[1])
      return numpy.nan
    return torch.sqrt(self.eigenValues[1]) * 4


def getLeastAxisLengthFeatureValue(self):
    r"""
    **15. Least Axis Length**

    .. math::
      \textit{least axis} = 4 \sqrt{\lambda_{least}}

    This feature yield the smallest axis length of the ROI-enclosing ellipsoid and is calculated using the largest
    principal component :math:`\lambda_{least}`. In case of a 2D segmentation, this value will be 0.

    The principal component analysis is performed using the physical coordinates of the voxel centers defining the ROI.
    It therefore takes spacing into account, but does not make use of the shape mesh.
    """
    if self.eigenValues[0] < 0:
      self.logger.warning('Least axis eigenvalue negative! (%g)', self.eigenValues[0])
      return numpy.nan
    return torch.sqrt(self.eigenValues[0]) * 4


def getElongationFeatureValue(self):
    r"""
    **16. Elongation**

    Elongation shows the relationship between the two largest principal components in the ROI shape.
    For computational reasons, this feature is defined as the inverse of true elongation.

    .. math::
      \textit{elongation} = \sqrt{\frac{\lambda_{minor}}{\lambda_{major}}}

    Here, :math:`\lambda_{\text{major}}` and :math:`\lambda_{\text{minor}}` are the lengths of the largest and second
    largest principal component axes. The values range between 1 (where the cross section through the first and second
    largest principal moments is circle-like (non-elongated)) and 0 (where the object is a maximally elongated: i.e. a 1
    dimensional line).

    The principal component analysis is performed using the physical coordinates of the voxel centers defining the ROI.
    It therefore takes spacing into account, but does not make use of the shape mesh.
    """
    if self.eigenValues[1] < 0 or self.eigenValues[2] < 0:
      self.logger.warning('Elongation eigenvalue negative! (%g, %g)', self.eigenValues[1], self.eigenValues[2])
      return numpy.nan
    return torch.sqrt(self.eigenValues[1] / self.eigenValues[2])


def getFlatnessFeatureValue(self):
    r"""
    **17. Flatness**

    Flatness shows the relationship between the largest and smallest principal components in the ROI shape.
    For computational reasons, this feature is defined as the inverse of true flatness.

    .. math::
      \textit{flatness} = \sqrt{\frac{\lambda_{least}}{\lambda_{major}}}

    Here, :math:`\lambda_{\text{major}}` and :math:`\lambda_{\text{least}}` are the lengths of the largest and smallest
    principal component axes. The values range between 1 (non-flat, sphere-like) and 0 (a flat object, or single-slice
    segmentation).

    The principal component analysis is performed using the physical coordinates of the voxel centers defining the ROI.
    It therefore takes spacing into account, but does not make use of the shape mesh.
    """
    if self.eigenValues[0] < 0 or self.eigenValues[2] < 0:
      self.logger.warning('Elongation eigenvalue negative! (%g, %g)', self.eigenValues[0], self.eigenValues[2])
      return numpy.nan
    return torch.sqrt(self.eigenValues[0] / self.eigenValues[2])
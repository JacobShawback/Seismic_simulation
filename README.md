# Seismic_simulation
For seismic simulation class 2021

## Explanation of files
- myPack/response_method.py
  - This file describes time integration, such as explicit method and Newmark-beta.
- myPack/make_matrix.py
  - Class 'House' is for making M, C, K matrix, and saving parameters of structure including M, C, K. (***Linear***)
  - Class 'House_NL' is for ***None Linear***.
- myPack/constitution.py
  - This file describes the constitution model (model of strain-stress).
  - Linear, Slip and Bilinear are constitution models.
  - Class 'Combined' is for combining constitution models in MDOF.

## How to use the constitution model
- The constitution models are initialized at myPack/response_method.py/NL_4dof/get_model().

- 'model' is an instance of Class 'Combined' in myPack/constitution.py.

- Use:

  - ```python
    x = self.dis_to_x(Dis)
    ```

    - x: elongation of springs (vector), Dis: displacement (vector)

  - ```python
    f = model.sheer(x)
    ```

    - f : reaction force (vector), x : elongation of springs (vector)

  - ```python
    F = self.x_to_dis(f)
    ```

    - F: internal force (vector), f : reaction force (vector)

- ***Be aware that (Dis, F) and (x, f) are written in different coordinates.***


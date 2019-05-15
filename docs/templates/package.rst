{{ fullname }} package
{% for item in range(8 + fullname|length) -%}={%- endfor %}

.. automodule:: {{ fullname }}
    {% if members -%}
    :members: {{ members|join(", ") }}
    :undoc-members:
    :show-inheritance:
    {%- endif %}

{% if submodules %}
    .. toctree::
       :maxdepth: 1
{% for item in submodules %}
       {{ fullname }}.{{ item }}
       {%- endfor %}
    {%- endif -%}

{% if subpackages %}
    .. toctree::
       :maxdepth: 1
{% for item in subpackages %}
       {{ fullname }}.{{ item }}
       {%- endfor %}
    {%- endif %}

{% set all = get_members(in_list='__all__', include_imported=True) %}

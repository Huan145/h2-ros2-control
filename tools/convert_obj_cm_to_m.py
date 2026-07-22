#!/usr/bin/env python3
"""
Converts an .obj mesh file from centimetres to metres by scaling all
vertex coordinates by 0.01.

Usage:
    python3 convert_obj_cm_to_m.py

Output:
    KLT_box_metres.obj  (in the same directory as the input file)
"""

INPUT_OBJ  = '/home/apt-ipc/Downloads/KLT_box/KLT_box.obj'
OUTPUT_OBJ = '/home/apt-ipc/Downloads/KLT_box/KLT_box_metres.obj'
SCALE      = 0.01   # cm → m


def convert(input_path, output_path, scale):
    vertices_converted = 0
    lines_written = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            stripped = line.strip()

            # Vertex line: "v x y z" or "v x y z w"
            if stripped.startswith('v '):
                parts = stripped.split()
                # parts[0] = 'v', parts[1..3] = x y z, parts[4] = optional w
                try:
                    x = float(parts[1]) * scale
                    y = float(parts[2]) * scale
                    z = float(parts[3]) * scale
                    if len(parts) == 5:
                        fout.write(f'v {x} {y} {z} {parts[4]}\n')
                    else:
                        fout.write(f'v {x} {y} {z}\n')
                    vertices_converted += 1
                except (ValueError, IndexError):
                    # Malformed vertex line — write as-is
                    fout.write(line)

            else:
                # All other lines (normals, UVs, faces, mtl refs etc) — unchanged
                fout.write(line)

            lines_written += 1

    print(f'Done!')
    print(f'  Input:  {input_path}')
    print(f'  Output: {output_path}')
    print(f'  Vertices converted: {vertices_converted}')
    print(f'  Scale factor: {scale} (cm → m)')


if __name__ == '__main__':
    convert(INPUT_OBJ, OUTPUT_OBJ, SCALE)

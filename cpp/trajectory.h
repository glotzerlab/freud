#ifndef _TRAJECTORY_H__
#define _TRAJECTORY_H__

//! Stores box dimensions and provides common routines for wrapping vectors back into the box
class Box
    {
    public:
        Box(float L) : m_Lx(l=L), m_Ly(L), m_Lz(L)
            {
            }
        Box(float Lx, float Ly, float Lz) : m_Lx(Lx), m_Ly(Ly), m_Lz(Lz) 
            {
            }
    
        float getLx()
            {
            return m_Lx;
            }
        float getLy()
            {
            return m_Ly;
            }
        float getLz()
            {
            return m_Lz;
            }
       
        void wrap(float &x, float &y, float &z)
            {
            x -= box.Lx * rintf(x * box.Lxinv);
            y -= box.Ly * rintf(y * box.Lyinv);
            z -= box.Lz * rintf(z * box.Lzinv);
            }       

    private:
        void setup()
            {
            m_Lx_inv = 1.0f / m_Lx;
            m_Ly_inv = 1.0f / m_Ly;
            m_Lz_inv = 1.0f / m_Lz;
            }
        float m_Lx, m_Ly, m_Lz;
        float m_Lx_inv, m_Ly_inv, m_Lz_inv;
    }


void export_trajectory();

#endif // _TRAJECTORY_H__


